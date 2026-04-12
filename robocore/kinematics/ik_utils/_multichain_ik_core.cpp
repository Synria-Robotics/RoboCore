// SPDX-License-Identifier: GPL-3.0-or-later
// Multi-end-effector DLS IK in unified configuration space (Eigen + pybind11).
// Matches Python _solve_multichain_ik: stacked 6K error/Jacobian, fixed damping DLS, step clip, q clamp.

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

enum class JointType : std::int32_t { Fixed = 0, Revolute = 1, Prismatic = 2 };

static Eigen::Matrix4d rodrigues_motion(const Eigen::Vector3d &axis_raw, double angle) {
  Eigen::Vector3d k = axis_raw;
  double n = k.norm();
  if (n < 1e-10) {
    return Eigen::Matrix4d::Identity();
  }
  k /= n;
  const double x = k[0], y = k[1], z = k[2];
  Eigen::Matrix3d K;
  K << 0, -z, y, z, 0, -x, -y, x, 0;
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  const Eigen::Matrix3d R =
      Eigen::Matrix3d::Identity() + s * K + (1.0 - c) * K * K;
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = R;
  return T;
}

static Eigen::Matrix4d prismatic_motion(const Eigen::Vector3d &axis_raw, double q) {
  Eigen::Vector3d a = axis_raw;
  double nn = a.norm();
  if (nn > 1e-10) {
    a /= nn;
  }
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 1>(0, 3) = a * q;
  return T;
}

/// R_err = R_tgt * R_cur^T, axis-angle vector (matches single-chain IK / Python multichain orientation error).
static Eigen::Vector3d rotation_error_compact(const Eigen::Matrix3d &R_cur, const Eigen::Matrix3d &R_tgt) {
  Eigen::Matrix3d R = R_tgt * R_cur.transpose();
  Eigen::AngleAxisd aa(R);
  return aa.axis() * aa.angle();
}

struct KinChain {
  std::vector<JointType> types_;
  std::vector<std::int32_t> q_idx_;
  std::vector<Eigen::Matrix4d> T_origin_;
  std::vector<Eigen::Vector3d> axes_;
};

static Eigen::Matrix4d fk_end_chain(const KinChain &ch, const Eigen::VectorXd &q) {
  const size_t n_joints = ch.types_.size();
  Eigen::Matrix4d T_parent = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d end_T = Eigen::Matrix4d::Identity();
  for (size_t ji = 0; ji < n_joints; ++ji) {
    const Eigen::Matrix4d T_joint_origin = T_parent * ch.T_origin_[ji];
    const std::int32_t qix = ch.q_idx_[ji];
    const JointType ty = ch.types_[ji];
    double qi = 0.0;
    if (qix >= 0) {
      qi = q[qix];
    }
    Eigen::Matrix4d Tm;
    switch (ty) {
    case JointType::Fixed:
      Tm = Eigen::Matrix4d::Identity();
      break;
    case JointType::Revolute:
      Tm = rodrigues_motion(ch.axes_[ji], qi);
      break;
    case JointType::Prismatic:
      Tm = prismatic_motion(ch.axes_[ji], qi);
      break;
    default:
      Tm = Eigen::Matrix4d::Identity();
      break;
    }
    const Eigen::Matrix4d T_child = T_joint_origin * Tm;
    T_parent = T_child;
    end_T = T_child;
  }
  return end_T;
}

static Eigen::MatrixXd jacobian_chain(const KinChain &ch, const Eigen::VectorXd &q, std::int32_t nq) {
  const size_t n_joints = ch.types_.size();
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, nq);

  std::vector<Eigen::Vector3d> z_col(static_cast<size_t>(nq));
  std::vector<Eigen::Vector3d> p_col(static_cast<size_t>(nq));
  std::vector<std::uint8_t> col_kind(static_cast<size_t>(nq), 0);
  std::vector<char> have(static_cast<size_t>(nq), 0);

  Eigen::Matrix4d T_parent = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d end_T = Eigen::Matrix4d::Identity();

  for (size_t ji = 0; ji < n_joints; ++ji) {
    const Eigen::Matrix4d T_joint_origin = T_parent * ch.T_origin_[ji];
    const std::int32_t qix = ch.q_idx_[ji];
    const JointType ty = ch.types_[ji];

    if (qix >= 0 && (ty == JointType::Revolute || ty == JointType::Prismatic)) {
      Eigen::Vector3d axis = ch.axes_[ji];
      double nn = axis.norm();
      if (nn > 1e-10) {
        axis /= nn;
      }
      const Eigen::Matrix3d Rjo = T_joint_origin.block<3, 3>(0, 0);
      z_col[static_cast<size_t>(qix)] = Rjo * axis;
      p_col[static_cast<size_t>(qix)] = T_joint_origin.block<3, 1>(0, 3);
      col_kind[static_cast<size_t>(qix)] =
          (ty == JointType::Revolute) ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(2);
      have[static_cast<size_t>(qix)] = 1;
    }

    double qi = 0.0;
    if (qix >= 0) {
      qi = q[qix];
    }
    Eigen::Matrix4d Tm;
    switch (ty) {
    case JointType::Fixed:
      Tm = Eigen::Matrix4d::Identity();
      break;
    case JointType::Revolute:
      Tm = rodrigues_motion(ch.axes_[ji], qi);
      break;
    case JointType::Prismatic:
      Tm = prismatic_motion(ch.axes_[ji], qi);
      break;
    default:
      Tm = Eigen::Matrix4d::Identity();
      break;
    }
    const Eigen::Matrix4d T_child = T_joint_origin * Tm;
    T_parent = T_child;
    end_T = T_child;
  }

  const Eigen::Vector3d p_end = end_T.block<3, 1>(0, 3);
  for (std::int32_t c = 0; c < nq; ++c) {
    const size_t ci = static_cast<size_t>(c);
    if (!have[ci]) {
      continue;
    }
    const Eigen::Vector3d &zv = z_col[ci];
    if (col_kind[ci] == 1) {
      J.col(c).head<3>() = zv.cross(p_end - p_col[ci]);
      J.col(c).tail<3>() = zv;
    } else {
      J.col(c).head<3>() = zv;
    }
  }
  return J;
}

static void clamp_q(Eigen::VectorXd &q, const Eigen::VectorXd &q_lo, const Eigen::VectorXd &q_hi,
                  const std::vector<std::uint8_t> &has_lo, const std::vector<std::uint8_t> &has_hi) {
  const std::int32_t n = static_cast<std::int32_t>(q.size());
  for (std::int32_t i = 0; i < n; ++i) {
    const size_t si = static_cast<size_t>(i);
    if (has_lo[si]) {
      q[i] = std::max(q[i], q_lo[i]);
    }
    if (has_hi[si]) {
      q[i] = std::min(q[i], q_hi[i]);
    }
  }
}

static py::dict solve_multichain_dls_impl(
    py::array_t<double> q0, py::array_t<double> targets, py::array_t<std::int32_t> joint_counts,
    py::array_t<std::int32_t> types_flat, py::array_t<std::int32_t> q_idx_flat,
    py::array_t<double> T_origin_flat, py::array_t<double> axes_flat, std::int32_t nq,
    std::int32_t max_iters, double pos_tol, double ori_tol, double damping, double step_limit,
    py::array_t<double> q_lo, py::array_t<double> q_hi, py::array_t<std::int32_t> has_lo,
    py::array_t<std::int32_t> has_hi) {

  py::buffer_info q0b = q0.request();
  py::buffer_info tt = targets.request();
  py::buffer_info jc = joint_counts.request();
  if (q0b.ndim != 1 || static_cast<std::int32_t>(q0b.shape[0]) != nq) {
    throw std::runtime_error("q0 must be (nq,) matching nq");
  }
  if (tt.ndim != 3 || tt.shape[1] != 4 || tt.shape[2] != 4) {
    throw std::runtime_error("targets must be (K, 4, 4)");
  }
  const ssize_t K = tt.shape[0];
  if (K < 1) {
    throw std::runtime_error("need at least one chain target");
  }
  if (jc.ndim != 1 || jc.shape[0] != K) {
    throw std::runtime_error("joint_counts must be (K,)");
  }

  py::buffer_info tf = types_flat.request();
  py::buffer_info qf = q_idx_flat.request();
  py::buffer_info tof = T_origin_flat.request();
  py::buffer_info axf = axes_flat.request();
  auto *jc_ptr = static_cast<const std::int32_t *>(jc.ptr);
  ssize_t sum_j = 0;
  for (ssize_t k = 0; k < K; ++k) {
    sum_j += jc_ptr[k];
  }
  if (tf.shape[0] != sum_j || qf.shape[0] != sum_j || tof.shape[0] != sum_j || tof.shape[1] != 16 ||
      axf.shape[0] != sum_j || axf.shape[1] != 3) {
    throw std::runtime_error("flat joint arrays length must equal sum(joint_counts)");
  }

  std::vector<KinChain> chains(static_cast<size_t>(K));
  auto *tp = static_cast<const std::int32_t *>(tf.ptr);
  auto *qp = static_cast<const std::int32_t *>(qf.ptr);
  auto *op = static_cast<const double *>(tof.ptr);
  auto *ap = static_cast<const double *>(axf.ptr);
  ssize_t off = 0;
  for (ssize_t k = 0; k < K; ++k) {
    const ssize_t nj = jc_ptr[k];
    KinChain &ch = chains[static_cast<size_t>(k)];
    ch.types_.resize(static_cast<size_t>(nj));
    ch.q_idx_.resize(static_cast<size_t>(nj));
    ch.T_origin_.resize(static_cast<size_t>(nj));
    ch.axes_.resize(static_cast<size_t>(nj));
    for (ssize_t j = 0; j < nj; ++j) {
      const ssize_t idx = off + j;
      ch.types_[static_cast<size_t>(j)] = static_cast<JointType>(tp[idx]);
      ch.q_idx_[static_cast<size_t>(j)] = qp[idx];
      Eigen::Map<const Eigen::Matrix4d> M(op + idx * 16);
      ch.T_origin_[static_cast<size_t>(j)] = M;
      ch.axes_[static_cast<size_t>(j)] =
          Eigen::Vector3d(ap[idx * 3 + 0], ap[idx * 3 + 1], ap[idx * 3 + 2]);
    }
    off += nj;
  }

  Eigen::VectorXd q = Eigen::Map<const Eigen::VectorXd>(static_cast<const double *>(q0b.ptr), nq);

  py::buffer_info blo = q_lo.request();
  py::buffer_info bhi = q_hi.request();
  py::buffer_info hlo = has_lo.request();
  py::buffer_info hhi = has_hi.request();
  if (blo.ndim != 1 || static_cast<std::int32_t>(blo.shape[0]) != nq || bhi.ndim != 1 ||
      static_cast<std::int32_t>(bhi.shape[0]) != nq) {
    throw std::runtime_error("q_lo and q_hi must be length nq");
  }
  if (hlo.ndim != 1 || static_cast<std::int32_t>(hlo.shape[0]) != nq || hhi.ndim != 1 ||
      static_cast<std::int32_t>(hhi.shape[0]) != nq) {
    throw std::runtime_error("has_lo and has_hi must be length nq");
  }
  Eigen::VectorXd q_lo_v = Eigen::Map<const Eigen::VectorXd>(static_cast<const double *>(blo.ptr), nq);
  Eigen::VectorXd q_hi_v = Eigen::Map<const Eigen::VectorXd>(static_cast<const double *>(bhi.ptr), nq);
  std::vector<std::uint8_t> has_lo_v(static_cast<size_t>(nq));
  std::vector<std::uint8_t> has_hi_v(static_cast<size_t>(nq));
  auto *pl = static_cast<const std::int32_t *>(hlo.ptr);
  auto *ph = static_cast<const std::int32_t *>(hhi.ptr);
  for (std::int32_t i = 0; i < nq; ++i) {
    has_lo_v[static_cast<size_t>(i)] = pl[i] != 0 ? 1 : 0;
    has_hi_v[static_cast<size_t>(i)] = ph[i] != 0 ? 1 : 0;
  }

  const int m = static_cast<int>(6 * K);
  Eigen::MatrixXd J(m, nq);
  Eigen::VectorXd e(m);
  Eigen::MatrixXd T_targets(static_cast<Eigen::Index>(K), 16);
  {
    auto *tptr = static_cast<const double *>(tt.ptr);
    const ssize_t s0 = tt.strides[0];
    const ssize_t s1 = tt.strides[1];
    const ssize_t s2 = tt.strides[2];
    for (ssize_t k = 0; k < K; ++k) {
      for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
          const char *p = reinterpret_cast<const char *>(tptr) + k * s0 + r * s1 + c * s2;
          T_targets(static_cast<Eigen::Index>(k), r * 4 + c) = *reinterpret_cast<const double *>(p);
        }
      }
    }
  }

  Eigen::VectorXd e_total;
  bool success = false;
  std::int32_t it_done = max_iters;

  for (std::int32_t it = 0; it < max_iters; ++it) {
    int row = 0;
    for (ssize_t k = 0; k < K; ++k) {
      Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> T_tgt(T_targets.row(static_cast<Eigen::Index>(k)).data());
      Eigen::Matrix4d T_cur = fk_end_chain(chains[static_cast<size_t>(k)], q);
      Eigen::Matrix3d R_cur = T_cur.block<3, 3>(0, 0);
      Eigen::Matrix3d R_tgt = T_tgt.block<3, 3>(0, 0);
      e.segment<3>(row) = T_tgt.block<3, 1>(0, 3) - T_cur.block<3, 1>(0, 3);
      e.segment<3>(row + 3) = rotation_error_compact(R_cur, R_tgt);
      J.block(row, 0, 6, nq) = jacobian_chain(chains[static_cast<size_t>(k)], q, nq);
      row += 6;
    }
    e_total = e;
    const double err_norm = e.norm();
    if (err_norm < pos_tol + ori_tol) {
      success = true;
      it_done = it + 1;
      break;
    }

    Eigen::MatrixXd A = J * J.transpose();
    const double lam2 = damping * damping;
    for (int i = 0; i < m; ++i) {
      A(i, i) += lam2;
    }
    Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
    Eigen::VectorXd y = ldlt.solve(e);
    Eigen::VectorXd dq = J.transpose() * y;

    double nd = dq.norm();
    if (step_limit > 0.0 && nd > step_limit) {
      dq *= step_limit / (nd + 1e-15);
    }
    q = q + dq;
    clamp_q(q, q_lo_v, q_hi_v, has_lo_v, has_hi_v);
  }

  // Match Python _solve_multichain_ik reporting (slice stacked 6K error vector).
  const int split = 3 * static_cast<int>(K);
  const double pos_err = e_total.head(split).norm();
  const double ori_err = e_total.tail(m - split).norm();

  py::list q_py;
  for (std::int32_t i = 0; i < nq; ++i) {
    q_py.append(q[i]);
  }
  py::dict out;
  out["q"] = q_py;
  out["success"] = success;
  out["iters"] = it_done;
  out["pos_err"] = pos_err;
  out["ori_err"] = ori_err;
  return out;
}

PYBIND11_MODULE(_multichain_ik_core, m) {
  m.doc() = "Multi-chain DLS IK in unified q (stacked tasks), Eigen.";
  m.def("solve_multichain_dls", &solve_multichain_dls_impl, py::arg("q0"), py::arg("targets"),
        py::arg("joint_counts"), py::arg("types_flat"), py::arg("q_idx_flat"), py::arg("T_origin_flat"),
        py::arg("axes_flat"), py::arg("nq"), py::arg("max_iters") = 200, py::arg("pos_tol") = 1e-3,
        py::arg("ori_tol") = 1e-3, py::arg("damping") = 1e-3, py::arg("step_limit") = 0.2,
        py::arg("q_lo"), py::arg("q_hi"), py::arg("has_lo"), py::arg("has_hi"));
}
