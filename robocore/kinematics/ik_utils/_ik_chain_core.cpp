// SPDX-License-Identifier: MIT
// Single-sample DLS IK (analytic Jacobian), Eigen + pybind11.
// Matches IKSolverNumPy._solve_single core path: DLS, adaptive damping/step, joint clamp.
// Does not implement Jacobian limit projection or limit-jump heuristics (see ik_solver_cpp docstring).

#ifdef _MSC_VER
#define _USE_MATH_DEFINES  // enable M_PI on MSVC
#endif
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;
using ssize_t = py::ssize_t;

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

/// rotation_error: axis*angle from R_current to R_target (world frame).
/// Stable SO(3) log map — handles small-angle and near-180° singularities without AngleAxisd.
static Eigen::Vector3d rotation_error_compact(const Eigen::Matrix3d &R_cur,
                                              const Eigen::Matrix3d &R_tgt) {
  const Eigen::Matrix3d R = R_tgt * R_cur.transpose();
  const double cos_angle = std::max(-1.0, std::min(1.0, (R.trace() - 1.0) * 0.5));
  const double angle = std::acos(cos_angle);
  const Eigen::Vector3d skew(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
  if (angle < 1e-10) {
    return skew * 0.5;  // first-order approximation for small angles
  }
  if (angle > M_PI - 1e-4) {
    // Near 180°: AngleAxisd is numerically unstable here. Extract axis from
    // R ≈ 2*n*n^T - I  =>  R(i,i)+1 = 2*n_i^2. Pick largest diagonal.
    int i = 0;
    if (R(1, 1) > R(0, 0)) i = 1;
    if (R(2, 2) > R(i, i)) i = 2;
    const double ni_sq = (R(i, i) + 1.0) * 0.5;
    Eigen::Vector3d axis = Eigen::Vector3d::Zero();
    if (ni_sq > 1e-20) {
      axis[i] = std::sqrt(ni_sq);
      const int j = (i + 1) % 3;
      const int k = (i + 2) % 3;
      axis[j] = (R(i, j) + R(j, i)) * 0.25 / axis[i];
      axis[k] = (R(i, k) + R(k, i)) * 0.25 / axis[i];
    } else {
      axis = Eigen::Vector3d::UnitX();
    }
    return axis.normalized() * angle;
  }
  return skew * (angle / (2.0 * std::sin(angle)));  // standard SO(3) log map
}

static Eigen::VectorXd solve_dls(const Eigen::MatrixXd &J, const Eigen::VectorXd &e, double lam) {
  Eigen::Matrix<double, 6, 6> A = J * J.transpose();
  A.diagonal().array() += lam * lam;
  Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(A);
  Eigen::VectorXd y = ldlt.solve(e);
  return J.transpose() * y;
}

/// Gauss-Newton of ‖J dq − e‖² + λ²‖dq‖² + Σ w_i (q_i+dq_i − q_prev_i)².
/// When every w_i ≤ 0 this is exactly ``solve_dls``. Per-joint w is the
/// Mink/Pink PostureTask: it uniquely selects the spherical-wrist nullspace
/// (j3↔j5 trade when j4≈0) without uniformly fighting the 6D task.
static Eigen::VectorXd solve_dls_smooth(const Eigen::MatrixXd &J, const Eigen::VectorXd &e, double lam,
                                        const Eigen::VectorXd &w, const Eigen::VectorXd &q,
                                        const Eigen::VectorXd &q_prev) {
  const int n = static_cast<int>(J.cols());
  if (w.size() != n || w.maxCoeff() <= 0.0) {
    return solve_dls(J, e, lam);
  }
  Eigen::MatrixXd A = J.transpose() * J;
  A.diagonal().array() += lam * lam + w.array();
  Eigen::VectorXd b = J.transpose() * e + w.cwiseProduct(q_prev - q);
  return A.ldlt().solve(b);
}

static Eigen::VectorXd parse_smooth_weight(const py::object &obj, int n) {
  Eigen::VectorXd w = Eigen::VectorXd::Zero(n);
  if (obj.is_none()) {
    return w;
  }
  if (py::isinstance<py::float_>(obj) || py::isinstance<py::int_>(obj)) {
    w.setConstant(obj.cast<double>());
    return w;
  }
  py::array_t<double> arr = py::array_t<double>::ensure(obj);
  py::buffer_info b = arr.request();
  if (b.ndim == 0 || (b.ndim == 1 && b.shape[0] == 1)) {
    w.setConstant(*static_cast<const double *>(b.ptr));
    return w;
  }
  if (b.ndim == 1 && static_cast<int>(b.shape[0]) == n) {
    const double *p = static_cast<const double *>(b.ptr);
    const ssize_t stride = (b.strides[0] == 0) ? 1 : b.strides[0] / static_cast<ssize_t>(sizeof(double));
    for (int i = 0; i < n; ++i) {
      w[i] = p[i * stride];
    }
    return w;
  }
  throw std::runtime_error("smooth_weight must be a scalar or a vector of n_dof");
}

/// Extra PostureTask on wrist roll/yaw when pitch (j4) is near 0.
/// s=1 at singularity, ~0 by |j4|≳3σ. Default σ=0.2 rad.
static void apply_wrist_singularity_lock(Eigen::VectorXd &w, const Eigen::VectorXd &q, double gain,
                                         double sigma = 0.2) {
  if (!(gain > 0.0) || w.size() < 6 || q.size() < 5) {
    return;
  }
  const double q4 = q[4];
  const double s = std::exp(-(q4 * q4) / (sigma * sigma));
  w[3] += gain * s;
  w[5] += gain * s;
}

static bool prior_active(const Eigen::VectorXd &w, double wrist_gain) {
  return (w.size() > 0 && w.maxCoeff() > 0.0) || (wrist_gain > 0.0);
}

static double adaptive_damping_value(const Eigen::MatrixXd &J, double pos_err, double ori_err,
                                     double min_d, double max_d) {
  // Cheap condition proxy: 6*||JJt||_F^2 / tr(JJt)^2 in [1,6].
  // 1 = uniform eigenvalues (well-conditioned), 6 = rank-1 (near-singular).
  // Replaces SelfAdjointEigenSolver (~O(6^3) per iteration) with O(36) ops.
  const Eigen::Matrix<double, 6, 6> JJt = J * J.transpose();
  const double tr = JJt.trace();
  double cond_proxy = 6.0;  // default: assume near-singular
  if (tr > 1e-20) {
    cond_proxy = 6.0 * JJt.squaredNorm() / (tr * tr);
  }
  const double err_combo = pos_err + 0.5 * ori_err;
  if (cond_proxy > 4.0 || err_combo > 0.05) {
    return max_d;
  }
  if (cond_proxy < 2.0 && err_combo < 0.01) {
    return min_d;
  }
  return std::sqrt(min_d * max_d);  // geometric mean
}

static double adaptive_step_value(double pos_err, double ori_err, double base_step) {
  double np = pos_err / 0.01;
  double no = ori_err / 0.087;
  double m = std::max(np, no);
  if (m > 2.0) {
    return base_step * 1.0;
  }
  if (m > 1.0) {
    return base_step * 1.2;
  }
  if (m > 0.5) {
    return base_step * 1.0;
  }
  return base_step * 0.5;
}

struct IkSolveOneResult {
  Eigen::VectorXd q;
  bool success;
  std::int32_t iters_done;
  double err_norm;
  double pos_err;
  double ori_err;
};

class ChainIKDls {
public:
  ChainIKDls(py::array_t<std::int32_t> joint_types, py::array_t<std::int32_t> q_indices,
             py::array_t<double> T_origin_colmajor, py::array_t<double> axes, std::int32_t n_dof,
             std::int32_t stop_chain_idx, py::array_t<double> q_limit_lo, py::array_t<double> q_limit_hi,
             py::array_t<std::int32_t> has_limit_lo, py::array_t<std::int32_t> has_limit_hi)
      : n_dof_(n_dof), stop_chain_idx_(stop_chain_idx) {
    py::buffer_info jt = joint_types.request();
    py::buffer_info qi = q_indices.request();
    py::buffer_info To = T_origin_colmajor.request();
    py::buffer_info ax = axes.request();
    const ssize_t n = jt.shape[0];
    if (qi.shape[0] != n || To.shape[0] != n || To.shape[1] != 16 || ax.shape[0] != n ||
        ax.shape[1] != 3) {
      throw std::runtime_error("ChainIKDls: inconsistent joint array shapes");
    }

    types_.resize(static_cast<size_t>(n));
    q_idx_.resize(static_cast<size_t>(n));
    T_origin_.resize(static_cast<size_t>(n));
    axes_.resize(static_cast<size_t>(n));

    auto *tp = static_cast<const std::int32_t *>(jt.ptr);
    auto *qp = static_cast<const std::int32_t *>(qi.ptr);
    auto *op = static_cast<const double *>(To.ptr);
    auto *ap = static_cast<const double *>(ax.ptr);

    for (ssize_t i = 0; i < n; ++i) {
      types_[static_cast<size_t>(i)] = static_cast<JointType>(tp[i]);
      q_idx_[static_cast<size_t>(i)] = qp[i];
      Eigen::Map<const Eigen::Matrix4d> M(op + i * 16);
      T_origin_[static_cast<size_t>(i)] = M;
      axes_[static_cast<size_t>(i)] =
          Eigen::Vector3d(ap[i * 3 + 0], ap[i * 3 + 1], ap[i * 3 + 2]);
    }

    py::buffer_info blo = q_limit_lo.request();
    py::buffer_info bhi = q_limit_hi.request();
    py::buffer_info hlo = has_limit_lo.request();
    py::buffer_info hhi = has_limit_hi.request();
    if (blo.ndim != 1 || static_cast<std::int32_t>(blo.shape[0]) != n_dof) {
      throw std::runtime_error("q_limit_lo length must match n_dof");
    }
    if (bhi.ndim != 1 || static_cast<std::int32_t>(bhi.shape[0]) != n_dof) {
      throw std::runtime_error("q_limit_hi length must match n_dof");
    }
    if (hlo.ndim != 1 || static_cast<std::int32_t>(hlo.shape[0]) != n_dof ||
        hhi.ndim != 1 || static_cast<std::int32_t>(hhi.shape[0]) != n_dof) {
      throw std::runtime_error("has_limit_* length must match n_dof");
    }
    q_lo_.resize(n_dof);
    q_hi_.resize(n_dof);
    std::memcpy(q_lo_.data(), static_cast<const double *>(blo.ptr), static_cast<size_t>(n_dof) * sizeof(double));
    std::memcpy(q_hi_.data(), static_cast<const double *>(bhi.ptr), static_cast<size_t>(n_dof) * sizeof(double));
    has_lo_.resize(static_cast<size_t>(n_dof));
    has_hi_.resize(static_cast<size_t>(n_dof));
    auto *plo = static_cast<const std::int32_t *>(hlo.ptr);
    auto *phi = static_cast<const std::int32_t *>(hhi.ptr);
    for (std::int32_t i = 0; i < n_dof; ++i) {
      has_lo_[static_cast<size_t>(i)] = plo[i] != 0;
      has_hi_[static_cast<size_t>(i)] = phi[i] != 0;
    }
  }

  IkSolveOneResult solve_one(const Eigen::Matrix4d &T_tgt, const Eigen::VectorXd &q0_in,
                             std::int32_t max_iters, double pos_tol, double ori_tol, double min_damping,
                             double max_damping, double base_step, double pos_weight, double ori_weight,
                             bool adaptive_damping, bool adaptive_step, double max_step_norm,
                             const Eigen::VectorXd &w_base, double wrist_gain,
                             const Eigen::VectorXd &q_prev_in) const {
    Eigen::Matrix3d R_tgt = T_tgt.block<3, 3>(0, 0);
    Eigen::Vector3d p_tgt = T_tgt.block<3, 1>(0, 3);

    Eigen::VectorXd q = q0_in;
    Eigen::VectorXd q_prev = q_prev_in;
    if (q_prev.size() != q.size()) {
      q_prev = q0_in;
    }
    Eigen::VectorXd best_q = q;
    double best_err = std::numeric_limits<double>::infinity();

    bool success = false;
    std::int32_t iters_done = max_iters;
    double prev_err = std::numeric_limits<double>::infinity();
    int plateau_count = 0;
    const int plateau_limit = 15;  // exit early if no meaningful improvement for this many iters

    for (std::int32_t it = 1; it <= max_iters; ++it) {
      // Single-pass FK + Jacobian: one chain traversal instead of two.
      std::pair<Eigen::Matrix4d, Eigen::MatrixXd> fk_j = fk_and_jacobian(q);
      const Eigen::Matrix4d &T_end = fk_j.first;
      Eigen::MatrixXd &J = fk_j.second;
      const Eigen::Matrix3d R_cur = T_end.block<3, 3>(0, 0);
      const Eigen::Vector3d p_cur = T_end.block<3, 1>(0, 3);

      const Eigen::Vector3d p_err = p_tgt - p_cur;
      const Eigen::Vector3d o_err = rotation_error_compact(R_cur, R_tgt);
      const double pos_norm = p_err.norm();
      const double ori_norm = o_err.norm();

      Eigen::VectorXd e(6);
      e.head<3>() = pos_weight * p_err;
      e.tail<3>() = ori_weight * o_err;
      const double err_norm = e.norm();

      if (err_norm < best_err) {
        best_err = err_norm;
        best_q = q;
      }

      // Plateau detection: exit early when no meaningful progress.
      if (prev_err - err_norm < 1e-8 * (1.0 + prev_err)) {
        if (++plateau_count >= plateau_limit) break;
      } else {
        plateau_count = 0;
      }
      prev_err = err_norm;

      if (pos_norm < pos_tol && ori_norm < ori_tol) {
        success = true;
        iters_done = it;
        best_err = err_norm;
        break;
      }

      J.topRows(3) *= pos_weight;
      J.bottomRows(3) *= ori_weight;

      double damp =
          adaptive_damping ? adaptive_damping_value(J, pos_norm, ori_norm, min_damping, max_damping)
                           : 0.5 * (min_damping + max_damping);

      Eigen::VectorXd w = w_base;
      apply_wrist_singularity_lock(w, q, wrist_gain);
      Eigen::VectorXd dq = solve_dls_smooth(J, e, damp, w, q, q_prev);

      double step = adaptive_step ? adaptive_step_value(pos_norm, ori_norm, base_step) : base_step;
      Eigen::VectorXd dq_step = dq * step;

      if (adaptive_step && max_step_norm > 0.0) {
        double nd = dq_step.norm();
        if (nd > max_step_norm) {
          dq_step *= max_step_norm / (nd + 1e-15);
        }
      }

      q = q + dq_step;
      clamp_q(q);
    }

    if (!success) {
      Eigen::Matrix4d T_best = fk_end(best_q);
      const double pos_best = (p_tgt - T_best.block<3, 1>(0, 3)).norm();
      const double ori_best = rotation_error_compact(T_best.block<3, 3>(0, 0), R_tgt).norm();
      const bool close = (pos_best <= 10.0 * pos_tol) && (ori_best <= 10.0 * ori_tol);
      if (prior_active(w_base, wrist_gain) && !close) {
        q = q0_in;
      } else {
        q = best_q;
      }
    }

    Eigen::Matrix4d T_final = fk_end(q);
    Eigen::Vector3d p_f = T_final.block<3, 1>(0, 3);
    Eigen::Matrix3d R_f = T_final.block<3, 3>(0, 0);
    double pos_err_final = (p_tgt - p_f).norm();
    double ori_err_final = rotation_error_compact(R_f, R_tgt).norm();

    IkSolveOneResult r;
    r.q = std::move(q);
    r.success = success;
    r.iters_done = iters_done;
    r.err_norm = best_err;
    r.pos_err = pos_err_final;
    r.ori_err = ori_err_final;
    return r;
  }

  py::dict solve(py::array_t<double> target_pose, py::array_t<double> q0, std::int32_t max_iters,
                 double pos_tol, double ori_tol, double min_damping, double max_damping, double base_step,
                 double pos_weight, double ori_weight, bool adaptive_damping, bool adaptive_step,
                 double max_step_norm, py::object smooth_weight_obj, py::object q_prev_obj,
                 double wrist_singularity_gain) const {
    py::buffer_info tp = target_pose.request();
    py::buffer_info q0b = q0.request();
    if (tp.ndim != 2 || tp.shape[0] != 4 || tp.shape[1] != 4) {
      throw std::runtime_error("target_pose must be (4, 4)");
    }
    if (q0b.ndim != 1 || static_cast<std::int32_t>(q0b.shape[0]) != n_dof_) {
      throw std::runtime_error("q0 must be (n_dof,)");
    }

    Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> T_tgt(
        static_cast<const double *>(tp.ptr));
    Eigen::VectorXd q_init =
        Eigen::Map<const Eigen::VectorXd>(static_cast<const double *>(q0b.ptr), n_dof_);
    Eigen::VectorXd q_pref = q_init;
    if (!q_prev_obj.is_none()) {
      py::array_t<double> qp = q_prev_obj.cast<py::array_t<double>>();
      py::buffer_info pb = qp.request();
      if (pb.ndim == 1 && static_cast<std::int32_t>(pb.shape[0]) == n_dof_) {
        q_pref = Eigen::Map<const Eigen::VectorXd>(static_cast<const double *>(pb.ptr), n_dof_);
      }
    }

    Eigen::VectorXd w_base = parse_smooth_weight(smooth_weight_obj, n_dof_);
    IkSolveOneResult sr = solve_one(T_tgt, q_init, max_iters, pos_tol, ori_tol, min_damping, max_damping,
                                    base_step, pos_weight, ori_weight, adaptive_damping, adaptive_step,
                                    max_step_norm, w_base, wrist_singularity_gain, q_pref);

    py::list q_py;
    for (std::int32_t i = 0; i < n_dof_; ++i) {
      q_py.append(sr.q[i]);
    }

    py::dict out;
    out["q"] = q_py;
    out["success"] = sr.success;
    out["iters"] = sr.iters_done;
    out["err_norm"] = sr.err_norm;
    out["pos_err"] = sr.pos_err;
    out["ori_err"] = sr.ori_err;
    out["method"] = std::string("dls");
    out["jacobian"] = std::string("analytic");
    return out;
  }

  /// B independent IK solves in one call (C++ loop; same numerics as ``solve``).
  py::dict solve_batch(py::array_t<double> target_poses, py::array_t<double> q0_batch,
                       std::int32_t max_iters, double pos_tol, double ori_tol, double min_damping,
                       double max_damping, double base_step, double pos_weight, double ori_weight,
                       bool adaptive_damping, bool adaptive_step, double max_step_norm,
                       py::object smooth_weight_obj, py::object q_prev_obj,
                       double wrist_singularity_gain) const {
    py::buffer_info tpb = target_poses.request();
    py::buffer_info q0b = q0_batch.request();
    if (tpb.ndim != 3 || tpb.shape[1] != 4 || tpb.shape[2] != 4) {
      throw std::runtime_error("target_poses must be (B, 4, 4)");
    }
    if (q0b.ndim != 2 || static_cast<std::int32_t>(q0b.shape[1]) != n_dof_) {
      throw std::runtime_error("q0_batch must be (B, n_dof)");
    }
    const ssize_t B = tpb.shape[0];
    if (q0b.shape[0] != B) {
      throw std::runtime_error("target_poses and q0_batch batch dimension mismatch");
    }
    if (B < 1) {
      throw std::runtime_error("batch size B must be >= 1");
    }

    std::vector<double> q_flat(static_cast<size_t>(B) * static_cast<size_t>(n_dof_));
    std::vector<std::uint8_t> succ(static_cast<size_t>(B));
    std::vector<std::int32_t> iters(static_cast<size_t>(B));
    std::vector<double> err_n(static_cast<size_t>(B));
    std::vector<double> pos_e(static_cast<size_t>(B));
    std::vector<double> ori_e(static_cast<size_t>(B));

    const double *tp = static_cast<const double *>(tpb.ptr);
    const double *qp = static_cast<const double *>(q0b.ptr);
    const ssize_t s0 = tpb.strides[0];
    const ssize_t s1 = tpb.strides[1];
    const ssize_t s2 = tpb.strides[2];
    const ssize_t qstride0 = q0b.strides[0];
    const ssize_t qstride1 = q0b.strides[1];

    Eigen::VectorXd w_base = parse_smooth_weight(smooth_weight_obj, n_dof_);

    bool have_prev_batch = false;
    bool have_prev_vec = false;
    py::array_t<double> q_prev_arr;
    py::buffer_info prevb;
    if (!q_prev_obj.is_none()) {
      q_prev_arr = q_prev_obj.cast<py::array_t<double>>();
      prevb = q_prev_arr.request();
      if (prevb.ndim == 2 && prevb.shape[0] == B && static_cast<std::int32_t>(prevb.shape[1]) == n_dof_) {
        have_prev_batch = true;
      } else if (prevb.ndim == 1 && static_cast<std::int32_t>(prevb.shape[0]) == n_dof_) {
        have_prev_vec = true;
      }
    }

    for (ssize_t b = 0; b < B; ++b) {
      Eigen::Matrix4d T_tgt;
      for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
          const char *row_ptr = reinterpret_cast<const char *>(tp) + b * s0 + r * s1 + c * s2;
          T_tgt(r, c) = *reinterpret_cast<const double *>(row_ptr);
        }
      }
      Eigen::VectorXd q_init(n_dof_);
      for (std::int32_t j = 0; j < n_dof_; ++j) {
        const char *qj =
            reinterpret_cast<const char *>(qp) + b * qstride0 + j * qstride1;
        q_init[j] = *reinterpret_cast<const double *>(qj);
      }

      Eigen::VectorXd q_pref = q_init;
      if (have_prev_batch) {
        for (std::int32_t j = 0; j < n_dof_; ++j) {
          const char *pj = reinterpret_cast<const char *>(prevb.ptr) + b * prevb.strides[0] +
                           j * prevb.strides[1];
          q_pref[j] = *reinterpret_cast<const double *>(pj);
        }
      } else if (have_prev_vec) {
        q_pref = Eigen::Map<const Eigen::VectorXd>(static_cast<const double *>(prevb.ptr), n_dof_);
      }

      IkSolveOneResult sr = solve_one(T_tgt, q_init, max_iters, pos_tol, ori_tol, min_damping,
                                      max_damping, base_step, pos_weight, ori_weight, adaptive_damping,
                                      adaptive_step, max_step_norm, w_base, wrist_singularity_gain, q_pref);
      const size_t bi = static_cast<size_t>(b);
      for (std::int32_t j = 0; j < n_dof_; ++j) {
        q_flat[bi * static_cast<size_t>(n_dof_) + static_cast<size_t>(j)] = sr.q[j];
      }
      succ[bi] = sr.success ? 1 : 0;
      iters[bi] = sr.iters_done;
      err_n[bi] = sr.err_norm;
      pos_e[bi] = sr.pos_err;
      ori_e[bi] = sr.ori_err;
    }

    py::array_t<double> q_arr(
        std::vector<ssize_t>{B, static_cast<ssize_t>(n_dof_)});
    std::memcpy(q_arr.mutable_data(), q_flat.data(), q_flat.size() * sizeof(double));

    py::array_t<std::uint8_t> succ_arr(std::vector<ssize_t>{B});
    auto *succ_w = succ_arr.mutable_data();
    for (ssize_t i = 0; i < B; ++i) {
      succ_w[i] = succ[static_cast<size_t>(i)];
    }

    py::array_t<std::int32_t> it_arr(std::vector<ssize_t>{B});
    std::memcpy(it_arr.mutable_data(), iters.data(), static_cast<size_t>(B) * sizeof(std::int32_t));

    py::array_t<double> en_arr(std::vector<ssize_t>{B});
    std::memcpy(en_arr.mutable_data(), err_n.data(), static_cast<size_t>(B) * sizeof(double));
    py::array_t<double> pe_arr(std::vector<ssize_t>{B});
    std::memcpy(pe_arr.mutable_data(), pos_e.data(), static_cast<size_t>(B) * sizeof(double));
    py::array_t<double> oe_arr(std::vector<ssize_t>{B});
    std::memcpy(oe_arr.mutable_data(), ori_e.data(), static_cast<size_t>(B) * sizeof(double));

    py::dict out;
    out["q"] = q_arr;
    out["success"] = succ_arr;
    out["iters"] = it_arr;
    out["err_norm"] = en_arr;
    out["pos_err"] = pe_arr;
    out["ori_err"] = oe_arr;
    out["method"] = py::str("dls");
    out["jacobian"] = py::str("analytic");
    return out;
  }

private:
  // Single-pass FK + Jacobian: computes end-effector pose and geometric Jacobian
  // in one chain traversal, replacing the previous fk_end() + jacobian() double-pass.
  std::pair<Eigen::Matrix4d, Eigen::MatrixXd> fk_and_jacobian(const Eigen::VectorXd &q) const {
    const size_t n_joints = types_.size();
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, n_dof_);
    std::vector<Eigen::Vector3d> z_col(static_cast<size_t>(n_dof_));
    std::vector<Eigen::Vector3d> p_col(static_cast<size_t>(n_dof_));
    std::vector<std::uint8_t> col_kind(static_cast<size_t>(n_dof_), 0);
    std::vector<char> have(static_cast<size_t>(n_dof_), 0);
    Eigen::Matrix4d T_parent = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d end_T = Eigen::Matrix4d::Identity();
    for (size_t ji = 0; ji < n_joints; ++ji) {
      const Eigen::Matrix4d T_joint_origin = T_parent * T_origin_[ji];
      const std::int32_t qix = q_idx_[ji];
      const JointType ty = types_[ji];
      if (qix >= 0 && (ty == JointType::Revolute || ty == JointType::Prismatic)) {
        Eigen::Vector3d axis = axes_[ji];
        const double nn = axis.norm();
        if (nn > 1e-10) axis /= nn;
        const Eigen::Matrix3d Rjo = T_joint_origin.block<3, 3>(0, 0);
        z_col[static_cast<size_t>(qix)] = Rjo * axis;
        p_col[static_cast<size_t>(qix)] = T_joint_origin.block<3, 1>(0, 3);
        col_kind[static_cast<size_t>(qix)] =
            (ty == JointType::Revolute) ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(2);
        have[static_cast<size_t>(qix)] = 1;
      }
      double qi = 0.0;
      if (qix >= 0) qi = q[qix];
      Eigen::Matrix4d Tm;
      switch (ty) {
      case JointType::Fixed:     Tm = Eigen::Matrix4d::Identity(); break;
      case JointType::Revolute:  Tm = rodrigues_motion(axes_[ji], qi); break;
      case JointType::Prismatic: Tm = prismatic_motion(axes_[ji], qi); break;
      default:                   Tm = Eigen::Matrix4d::Identity(); break;
      }
      const Eigen::Matrix4d T_child = T_joint_origin * Tm;
      T_parent = T_child;
      end_T = T_child;
      if (stop_chain_idx_ >= 0 && static_cast<std::int32_t>(ji) == stop_chain_idx_) break;
    }
    const Eigen::Vector3d p_end = end_T.block<3, 1>(0, 3);
    for (std::int32_t c = 0; c < n_dof_; ++c) {
      const size_t ci = static_cast<size_t>(c);
      if (!have[ci]) continue;
      const Eigen::Vector3d &zv = z_col[ci];
      if (col_kind[ci] == 1) {
        J.col(c).head<3>() = zv.cross(p_end - p_col[ci]);
        J.col(c).tail<3>() = zv;
      } else {
        J.col(c).head<3>() = zv;
      }
    }
    return {end_T, std::move(J)};
  }

  void clamp_q(Eigen::VectorXd &q) const {
    for (std::int32_t i = 0; i < n_dof_; ++i) {
      const size_t si = static_cast<size_t>(i);
      if (has_lo_[si]) {
        q[i] = std::max(q[i], q_lo_[i]);
      }
      if (has_hi_[si]) {
        q[i] = std::min(q[i], q_hi_[i]);
      }
    }
  }

  Eigen::Matrix4d fk_end(const Eigen::VectorXd &q) const {
    const size_t n_joints = types_.size();
    Eigen::Matrix4d T_parent = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d end_T = Eigen::Matrix4d::Identity();

    for (size_t ji = 0; ji < n_joints; ++ji) {
      const Eigen::Matrix4d T_joint_origin = T_parent * T_origin_[ji];
      const std::int32_t qix = q_idx_[ji];
      const JointType ty = types_[ji];
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
        Tm = rodrigues_motion(axes_[ji], qi);
        break;
      case JointType::Prismatic:
        Tm = prismatic_motion(axes_[ji], qi);
        break;
      default:
        Tm = Eigen::Matrix4d::Identity();
        break;
      }
      const Eigen::Matrix4d T_child = T_joint_origin * Tm;
      T_parent = T_child;
      end_T = T_child;
      if (stop_chain_idx_ >= 0 && static_cast<std::int32_t>(ji) == stop_chain_idx_) {
        break;
      }
    }
    return end_T;
  }

  Eigen::MatrixXd jacobian(const Eigen::VectorXd &q) const {
    const size_t n_joints = types_.size();
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, n_dof_);

    std::vector<Eigen::Vector3d> z_col(static_cast<size_t>(n_dof_));
    std::vector<Eigen::Vector3d> p_col(static_cast<size_t>(n_dof_));
    std::vector<std::uint8_t> col_kind(static_cast<size_t>(n_dof_), 0);
    std::vector<char> have(static_cast<size_t>(n_dof_), 0);

    Eigen::Matrix4d T_parent = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d end_T = Eigen::Matrix4d::Identity();

    for (size_t ji = 0; ji < n_joints; ++ji) {
      const Eigen::Matrix4d T_joint_origin = T_parent * T_origin_[ji];
      const std::int32_t qix = q_idx_[ji];
      const JointType ty = types_[ji];

      if (qix >= 0 && (ty == JointType::Revolute || ty == JointType::Prismatic)) {
        Eigen::Vector3d axis = axes_[ji];
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
        Tm = rodrigues_motion(axes_[ji], qi);
        break;
      case JointType::Prismatic:
        Tm = prismatic_motion(axes_[ji], qi);
        break;
      default:
        Tm = Eigen::Matrix4d::Identity();
        break;
      }
      const Eigen::Matrix4d T_child = T_joint_origin * Tm;
      T_parent = T_child;
      end_T = T_child;
      if (stop_chain_idx_ >= 0 && static_cast<std::int32_t>(ji) == stop_chain_idx_) {
        break;
      }
    }

    const Eigen::Vector3d p_end = end_T.block<3, 1>(0, 3);
    for (std::int32_t c = 0; c < n_dof_; ++c) {
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

  std::int32_t n_dof_;
  std::int32_t stop_chain_idx_;
  std::vector<JointType> types_;
  std::vector<std::int32_t> q_idx_;
  std::vector<Eigen::Matrix4d> T_origin_;
  std::vector<Eigen::Vector3d> axes_;
  Eigen::VectorXd q_lo_;
  Eigen::VectorXd q_hi_;
  std::vector<bool> has_lo_;
  std::vector<bool> has_hi_;
};

PYBIND11_MODULE(_ik_chain_core, m) {
  m.doc() = "DLS IK (analytic Jacobian), matches NumPy single-sample core (no limit Jacobian projection).";
  py::class_<ChainIKDls>(m, "ChainIKDls")
      .def(py::init<py::array_t<std::int32_t>, py::array_t<std::int32_t>, py::array_t<double>,
                     py::array_t<double>, std::int32_t, std::int32_t, py::array_t<double>,
                     py::array_t<double>, py::array_t<std::int32_t>, py::array_t<std::int32_t>>())
      .def("solve", &ChainIKDls::solve, py::arg("target_pose"), py::arg("q0"), py::arg("max_iters") = 200,
           py::arg("pos_tol") = 1e-3, py::arg("ori_tol") = 1e-3, py::arg("min_damping") = 1e-4,
           py::arg("max_damping") = 5e-2, py::arg("base_step") = 1.0, py::arg("pos_weight") = 1.0,
           py::arg("ori_weight") = 1.0, py::arg("adaptive_damping") = true, py::arg("adaptive_step") = true,
           py::arg("max_step_norm") = 0.5, py::arg("smooth_weight") = 0.0,
           py::arg("q_prev") = py::none(), py::arg("wrist_singularity_gain") = 0.0)
      .def("solve_batch", &ChainIKDls::solve_batch, py::arg("target_poses"), py::arg("q0_batch"),
           py::arg("max_iters") = 200, py::arg("pos_tol") = 1e-3, py::arg("ori_tol") = 1e-3,
           py::arg("min_damping") = 1e-4, py::arg("max_damping") = 5e-2, py::arg("base_step") = 1.0,
           py::arg("pos_weight") = 1.0, py::arg("ori_weight") = 1.0, py::arg("adaptive_damping") = true,
           py::arg("adaptive_step") = true, py::arg("max_step_norm") = 0.5,
           py::arg("smooth_weight") = 0.0, py::arg("q_prev") = py::none(),
           py::arg("wrist_singularity_gain") = 0.0);
}
