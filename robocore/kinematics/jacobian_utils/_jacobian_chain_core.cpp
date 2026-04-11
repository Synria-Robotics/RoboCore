// SPDX-License-Identifier: GPL-3.0-or-later
// Geometric Jacobian 6 x n_dof (world frame), Eigen + pybind11.

#include <Eigen/Dense>
#include <cstdint>
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
  double n = a.norm();
  if (n > 1e-10) {
    a /= n;
  }
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 1>(0, 3) = a * q;
  return T;
}

/// stop_chain_idx: inclusive joint index in chain to stop after (-1 = full chain to end).
class ChainJacobian {
public:
  ChainJacobian(py::array_t<std::int32_t> joint_types, py::array_t<std::int32_t> q_indices,
                py::array_t<double> T_origin_colmajor, py::array_t<double> axes,
                std::int32_t n_dof, std::int32_t stop_chain_idx)
      : n_dof_(n_dof), stop_chain_idx_(stop_chain_idx) {
    py::buffer_info jt = joint_types.request();
    py::buffer_info qi = q_indices.request();
    py::buffer_info To = T_origin_colmajor.request();
    py::buffer_info ax = axes.request();

    if (jt.ndim != 1 || qi.ndim != 1 || To.ndim != 2 || ax.ndim != 2) {
      throw std::runtime_error("invalid array ranks for ChainJacobian");
    }
    const ssize_t n = jt.shape[0];
    if (qi.shape[0] != n || To.shape[0] != n || To.shape[1] != 16 || ax.shape[0] != n ||
        ax.shape[1] != 3) {
      throw std::runtime_error("ChainJacobian: inconsistent joint array shapes");
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
  }

  py::array_t<double> jacobian_analytic(py::array_t<double> q) const {
    py::buffer_info qb = q.request();
    if (qb.ndim == 1) {
      if (static_cast<std::int32_t>(qb.shape[0]) != n_dof_) {
        throw std::runtime_error("q length does not match n_dof");
      }
      py::array_t<double> out(std::vector<ssize_t>{1, 6, n_dof_});
      jac_impl(static_cast<const double *>(qb.ptr), 1, static_cast<double *>(out.mutable_data()));
      return out;
    }
    if (qb.ndim != 2) {
      throw std::runtime_error("q must be (n_dof,) or (batch, n_dof)");
    }
    if (static_cast<std::int32_t>(qb.shape[1]) != n_dof_) {
      throw std::runtime_error("q.shape[1] does not match n_dof");
    }
    const ssize_t batch = qb.shape[0];
    py::array_t<double> out(std::vector<ssize_t>{batch, 6, n_dof_});
    jac_impl(static_cast<const double *>(qb.ptr), batch, static_cast<double *>(out.mutable_data()));
    return out;
  }

private:
  void jac_impl(const double *q_ptr, ssize_t batch, double *out_ptr) const {
    const size_t n_joints = types_.size();
    for (ssize_t b = 0; b < batch; ++b) {
      const double *qb = q_ptr + b * n_dof_;
      std::vector<Eigen::Vector3d> z_col(static_cast<size_t>(n_dof_));
      std::vector<Eigen::Vector3d> p_col(static_cast<size_t>(n_dof_));
      std::vector<std::uint8_t> col_kind(static_cast<size_t>(n_dof_), 0); // 1 rev, 2 prism
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
          qi = qb[qix];
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
          for (int r = 0; r < 6; ++r) {
            out_ptr[b * 6 * n_dof_ + r * n_dof_ + c] = 0.0;
          }
          continue;
        }
        const Eigen::Vector3d &z = z_col[ci];
        if (col_kind[ci] == 1) {
          const Eigen::Vector3d Jv = z.cross(p_end - p_col[ci]);
          for (int r = 0; r < 3; ++r) {
            out_ptr[b * 6 * n_dof_ + r * n_dof_ + c] = Jv[static_cast<Eigen::Index>(r)];
          }
          for (int r = 0; r < 3; ++r) {
            out_ptr[b * 6 * n_dof_ + (3 + r) * n_dof_ + c] = z[static_cast<Eigen::Index>(r)];
          }
        } else {
          for (int r = 0; r < 3; ++r) {
            out_ptr[b * 6 * n_dof_ + r * n_dof_ + c] = z[static_cast<Eigen::Index>(r)];
          }
          for (int r = 0; r < 3; ++r) {
            out_ptr[b * 6 * n_dof_ + (3 + r) * n_dof_ + c] = 0.0;
          }
        }
      }
    }
  }

  std::int32_t n_dof_;
  std::int32_t stop_chain_idx_;
  std::vector<JointType> types_;
  std::vector<std::int32_t> q_idx_;
  std::vector<Eigen::Matrix4d> T_origin_;
  std::vector<Eigen::Vector3d> axes_;
};

PYBIND11_MODULE(_jacobian_chain_core, m) {
  m.doc() = "Geometric Jacobian 6xn (world frame), matches JacobianSolverNumPy analytic.";
  py::class_<ChainJacobian>(m, "ChainJacobian")
      .def(py::init<py::array_t<std::int32_t>, py::array_t<std::int32_t>, py::array_t<double>,
                     py::array_t<double>, std::int32_t, std::int32_t>())
      .def("jacobian_analytic", &ChainJacobian::jacobian_analytic);
}
