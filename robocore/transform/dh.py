"""DH 参数提取工具

从 `RobotModel` 中提取近似的标准 DH (Denavit-Hartenberg) 与改进（Modified DH）参数。

说明：
1. 由于 URDF 采用通用 4x4 变换 (origin rpy + xyz) 串联，不一定严格符合 DH 规范；
2. 这里我们按“最小扭转”策略：逐关节取当前（默认零位）构型下相邻关节轴 z_i, z_{i+1} 与原点 O_i, O_{i+1} 来拟合；
3. 对于标准 DH：
   - α_i:  z_i 到 z_{i+1} 的夹角（绕 x_i）
   - a_i:  x_i 轴上 O_{i+1} 在当前帧的投影长度
   - d_i:  O_i 到 O_{i+1} 沿 z_i 的距离
   - θ_i:  关节变量（revolute 时为当前 q_i，prismatic 时为常量偏移）
4. Modified DH (Craig)：
   - α_{i-1}: z_{i-1} 到 z_i 角
   - a_{i-1}: x_{i-1} 到 x_i 间沿 x_{i-1} 距离
   - d_i:     O_{i-1} 到 O_i 沿 z_{i-1}
   - θ_i:     绕 z_{i-1} 的角度

注意：真实工业机器人若轴线并不严格相交/平行，拟合会引入残差；我们同时返回误差指标，供用户评估可接受性。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np

try:
	from robocore.modeling.robot_model import RobotModel
except Exception:  # pragma: no cover
	RobotModel = object  # type: ignore


@dataclass
class DHRow:
	alpha: float
	a: float
	d: float
	theta: float
	joint_name: str
	joint_type: str


@dataclass
class DHExtractionResult:
	standard: List[DHRow]
	modified: List[DHRow]
	axis_misalignment_norm: float  # 平均轴不共面 / 交差误差指标
	notes: str
	q_ref: List[float]


def _normalize(v: np.ndarray) -> np.ndarray:
	n = np.linalg.norm(v)
	if n < 1e-12:
		return v * 0.0
	return v / n


def _closest_points_between_lines(p1, d1, p2, d2) -> Tuple[np.ndarray, np.ndarray]:
	"""求两条空间直线(点+方向)的最近点对 (用于评估轴偏差)。"""
	d1 = _normalize(d1)
	d2 = _normalize(d2)
	w0 = p1 - p2
	a = d1.dot(d1)
	b = d1.dot(d2)
	c = d2.dot(d2)
	d = d1.dot(w0)
	e = d2.dot(w0)
	denom = a * c - b * b
	if abs(denom) < 1e-12:
		# 平行或几乎平行
		sc = 0.0
		tc = d / b if abs(b) > 1e-12 else 0.0
	else:
		sc = (b * e - c * d) / denom
		tc = (a * e - b * d) / denom
	pt1 = p1 + sc * d1
	pt2 = p2 + tc * d2
	return pt1, pt2


def extract_dh_parameters(model, q=None):
	"""从 RobotModel 拟合提取标准与改进 DH 参数（基于当前关节角 q, 默认为全 0）。

	返回的行数 = DOF（只处理 actuated joints）。
	对 prismatic 关节，θ 视为固定偏移，d 包含变量；对 revolute 关节反之。
	"""
	if q is None:
		q = [0.0] * model.dof()
	fk = model.forward_kinematics(q, return_numpy=True)
	end_key = "end"
	# 收集各关节原点与 z 轴方向（在世界系）
	joint_specs = model._actuated  # type: ignore[attr-defined]
	origins = []
	z_axes = []
	names = []
	types = []
	# 重建沿链路的逐关节 pose：通过 URDF 顺序 (model._chain_joints)
	# 建一个 name -> pose 映射
	poses = fk  # link 名到4x4
	# 对每个 joint origin：取其 parent link 的 pose * origin(rpy, xyz)
	import math
	def rpy(r,p,y):
		sr,cr = math.sin(r), math.cos(r)
		sp,cp = math.sin(p), math.cos(p)
		sy,cy = math.sin(y), math.cos(y)
		return np.array([
			[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
			[sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
			[-sp,   cp*sr,            cp*cr],
		])
	for js in model._chain_joints:  # type: ignore[attr-defined]
		# parent pose
		parent_pose = poses[js.parent]
		if not isinstance(parent_pose, np.ndarray):
			parent_pose = np.array(parent_pose)
		R_parent = parent_pose[:3,:3]
		t_parent = parent_pose[:3,3]
		R_o = rpy(*js.origin_rpy)
		t_o = np.array(js.origin_xyz)
		R_joint_origin = R_parent @ R_o
		p_joint_origin = t_parent + R_parent @ t_o
		if js.joint_type in ("revolute", "prismatic"):
			spec = next(s for s in joint_specs if s.name == js.name)
			axis_local = np.array(js.axis)
			axis_world = R_joint_origin @ axis_local
			origins.append(p_joint_origin)
			z_axes.append(_normalize(axis_world))
			names.append(js.name)
			types.append(js.joint_type)

	n = len(origins)
	origins = np.stack(origins, axis=0)
	z_axes = np.stack(z_axes, axis=0)

	# 评估相邻轴的“不相交”误差：对每对相邻求最近点距离
	mis_dists = []
	for i in range(n-1):
		p1, p2 = _closest_points_between_lines(origins[i], z_axes[i], origins[i+1], z_axes[i+1])
		mis_dists.append(np.linalg.norm(p1 - p2))
	axis_misalign = float(np.mean(mis_dists)) if mis_dists else 0.0

	standard_rows: List[DHRow] = []
	modified_rows: List[DHRow] = []

	# --- 标准 DH 拟合 ---
	# 参考：帧 i 的 z_i = 关节轴 i，x_i 选为 z_i 与 z_{i+1} 的最短公垂线方向（若平行则任取垂直）
	x_dirs = []
	for i in range(n-1):
		z_i = z_axes[i]
		z_next = z_axes[i+1]
		cross = np.cross(z_i, z_next)
		if np.linalg.norm(cross) < 1e-8:
			# 平行，选一个与 z_i 垂直的稳定方向
			tmp = np.array([1.0,0,0]) if abs(z_i[0]) < 0.9 else np.array([0,1.0,0])
			cross = np.cross(z_i, tmp)
		x_i = _normalize(np.cross(cross, z_i))  # 保证 x_i 垂直 z_i 且位于平面内
		x_dirs.append(x_i)
	# 最后一帧的 x_n 沿用前一个或任意正交向量
	if n>0:
		x_dirs.append(x_dirs[-1] if x_dirs else np.array([1,0,0]))

	for i in range(n):
		if i < n-1:
			z_i = z_axes[i]; z_next = z_axes[i+1]
			# α_i = angle between z_i and z_{i+1}
			cos_alpha = np.clip(z_i.dot(z_next), -1.0, 1.0)
			alpha = float(np.arccos(cos_alpha))
			# d_i = (O_{i+1} - O_i) 在 z_i 上的投影
			delta = origins[i+1] - origins[i]
			d_i = float(delta.dot(z_i))
			# a_i = (O_{i+1} - O_i) 在 x_i 上的投影
			a_i = float(delta.dot(x_dirs[i]))
		else:  # 最后一行用 0 / 0 处理（或根据末端工具再细化）
			alpha = 0.0; d_i = 0.0; a_i = 0.0
		# theta_i: 对 revolute 当前 q_i, prismatic 固定
		theta = float(q[i]) if types[i] == "revolute" else 0.0
		row = DHRow(alpha=alpha, a=a_i, d=d_i if types[i]=="revolute" else d_i+q[i], theta=theta, joint_name=names[i], joint_type=types[i])
		standard_rows.append(row)

	# --- Modified DH 拟合 (Craig) ---
	# 这里采用简单近似：
	# alpha_{i-1} = angle(z_{i-1}, z_i)
	# a_{i-1}     = (O_i - O_{i-1}) 在 x_{i-1} 上投影（复用上面 x_dirs）
	# d_i         = (O_i - O_{i-1}) 在 z_{i-1} 上投影
	# theta_i     = 绕 z_{i-1} 的旋转（取当前 q[i] 对 revolute / 0 对 prismatic）
	for i in range(n):
		if i == 0:
			alpha_prev = 0.0; a_prev = 0.0; d_i = 0.0
		else:
			z_prev = z_axes[i-1]; z_i = z_axes[i]
			cos_alpha = np.clip(z_prev.dot(z_i), -1.0, 1.0)
			alpha_prev = float(np.arccos(cos_alpha))
			delta = origins[i] - origins[i-1]
			a_prev = float(delta.dot(x_dirs[i-1]))
			d_i = float(delta.dot(z_prev))
		theta_i = float(q[i]) if types[i] == "revolute" else 0.0
		row_m = DHRow(alpha=alpha_prev, a=a_prev, d=d_i if types[i]=="revolute" else d_i+q[i], theta=theta_i, joint_name=names[i], joint_type=types[i])
		modified_rows.append(row_m)

	notes = (
		f"Extracted on zero/ given configuration; axis mean misalignment={axis_misalign:.3e} m. "
		"Values are approximate if robot is not strict DH chain."
	)

	return DHExtractionResult(
		standard=standard_rows,
		modified=modified_rows,
		axis_misalignment_norm=axis_misalign,
		notes=notes,
		q_ref=list(q),
	)


# ----------------- 基于提取 DH 重构 FK -----------------
def _dh_transform(alpha, a, d, theta):
	ca, sa = np.cos(alpha), np.sin(alpha)
	ct, st = np.cos(theta), np.sin(theta)
	return np.array([
		[ct, -st * ca, st * sa, a * ct],
		[st, ct * ca, -ct * sa, a * st],
		[0.0, sa, ca, d],
		[0.0, 0.0, 0.0, 1.0],
	])


def forward_kinematics_dh_standard(rows: List[DHRow], q: List[float]):
	"""使用提取的 *标准* DH 行与新的关节角 q 计算末端 4x4 变换。

	对 revolute: theta = q[i]; d = row.d
	对 prismatic: theta = row.theta(参考); d = row.d + q[i]
	"""
	T = np.eye(4)
	for i, r in enumerate(rows):
		if r.joint_type == "revolute":
			theta = q[i]
			d = r.d
		else:
			theta = r.theta
			d = r.d + q[i]
		A = _dh_transform(r.alpha, r.a, d, theta)
		T = T @ A
	return T


def forward_kinematics_dh_modified(rows: List[DHRow], q: List[float]):
	"""Craig Modified DH FK。行 i 存储 alpha_{i-1}, a_{i-1}, d_i, theta_i 基值。

	对 revolute: theta_i = q[i]; d_i = row.d
	对 prismatic: theta_i = row.theta; d_i = row.d + q[i]

	Modified DH 单步矩阵：A_i = Rot_z(theta_i) * Trans_z(d_i) * Trans_x(a_{i-1}) * Rot_x(alpha_{i-1})
	可写成等价的组合矩阵（此处直接构造）。
	"""
	T = np.eye(4)
	for i, r in enumerate(rows):
		if r.joint_type == "revolute":
			theta = q[i]
			d = r.d
		else:
			theta = r.theta
			d = r.d + q[i]
		ca, sa = np.cos(r.alpha), np.sin(r.alpha)
		ct, st = np.cos(theta), np.sin(theta)
		# 分解矩阵
		A = np.array([
			[ct, -st, 0.0, r.a],
			[st * ca, ct * ca, -sa, -sa * d],
			[st * sa, ct * sa, ca, ca * d],
			[0, 0, 0, 1],
		])
		# 上面写法容易混淆；为稳妥可以直接用分步乘法（更易理解）
		# 但先保持此形式，若出现偏差可再调整。
		T = T @ A
	return T


def _orientation_error_angle(Ra, Rb):
	tr = np.trace(Ra.T @ Rb)
	val = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
	return float(np.arccos(val))


def evaluate_dh_fit(model, dh_result: DHExtractionResult, samples=50, scale=0.6, seed=0):
	"""统计 DH 拟合 FK 与 原始 FK 的末端误差（位置 + 姿态角）。"""
	rng = np.random.default_rng(seed)
	# 采样关节 (在 joint limit 中心附近 scale 缩放)
	qs = []
	for _ in range(samples):
		q = [0.0] * model.dof()
		for js in model._actuated:  # type: ignore[attr-defined]
			lo, hi = -1.0, 1.0
			if js.limit:
				if js.limit[0] is not None: lo = js.limit[0]
				if js.limit[1] is not None: hi = js.limit[1]
			mid = 0.5 * (lo + hi)
			span = 0.5 * (hi - lo) * scale
			q[js.index] = float(rng.uniform(mid - span, mid + span))
		qs.append(q)

	pos_err_std = []
	ang_err_std = []
	pos_err_mod = []
	ang_err_mod = []

	for q in qs:
		fk = model.forward_kinematics(q, return_numpy=True)["end"]
		if not isinstance(fk, np.ndarray):
			fk = np.array(fk)
		p_ref = fk[:3, 3]; R_ref = fk[:3, :3]
		T_std = forward_kinematics_dh_standard(dh_result.standard, q)
		T_mod = forward_kinematics_dh_modified(dh_result.modified, q)
		p_std = T_std[:3, 3]; R_std = T_std[:3, :3]
		p_mod = T_mod[:3, 3]; R_mod = T_mod[:3, :3]
		pos_err_std.append(float(np.linalg.norm(p_std - p_ref)))
		pos_err_mod.append(float(np.linalg.norm(p_mod - p_ref)))
		ang_err_std.append(_orientation_error_angle(R_std, R_ref))
		ang_err_mod.append(_orientation_error_angle(R_mod, R_ref))

	def stats(x):
		return {
			'mean': float(np.mean(x)),
			'max': float(np.max(x)),
			'median': float(np.median(x)),
		}

	return {
		'standard_pos': stats(pos_err_std),
		'standard_ang': stats(ang_err_std),
		'modified_pos': stats(pos_err_mod),
		'modified_ang': stats(ang_err_mod),
		'samples': samples,
	}


__all__ = [
	"DHRow",
	"DHExtractionResult",
	"extract_dh_parameters",
	"forward_kinematics_dh_standard",
	"forward_kinematics_dh_modified",
	"evaluate_dh_fit",
	"forward_kinematics_dh_standard_all",
	"forward_kinematics_dh_modified_all",
]


def forward_kinematics_dh_standard_all(rows: List[DHRow], q: List[float]):
	"""返回标准 DH 每个关节后的累计变换列表 (长度 = len(rows))."""
	Ts = []
	T = np.eye(4)
	for i, r in enumerate(rows):
		if r.joint_type == "revolute":
			theta = q[i]; d = r.d
		else:
			theta = r.theta; d = r.d + q[i]
		A = _dh_transform(r.alpha, r.a, d, theta)
		T = T @ A
		Ts.append(T.copy())
	return Ts


def forward_kinematics_dh_modified_all(rows: List[DHRow], q: List[float]):
	"""返回改进 DH 每个关节后的累计变换列表 (长度 = len(rows))."""
	Ts = []
	T = np.eye(4)
	for i, r in enumerate(rows):
		if r.joint_type == "revolute":
			theta = q[i]; d = r.d
		else:
			theta = r.theta; d = r.d + q[i]
		ca, sa = np.cos(r.alpha), np.sin(r.alpha)
		ct, st = np.cos(theta), np.sin(theta)
		A = np.array([
			[ct, -st, 0.0, r.a],
			[st * ca, ct * ca, -sa, -sa * d],
			[st * sa, ct * sa, ca, ca * d],
			[0, 0, 0, 1],
		])
		T = T @ A
		Ts.append(T.copy())
	return Ts

