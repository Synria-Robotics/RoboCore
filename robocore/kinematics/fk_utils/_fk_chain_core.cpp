// SPDX-License-Identifier: MIT
// Serial-chain FK (revolute / prismatic / fixed), Eigen + pybind11.

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

class ChainFK {
public:
  /// T_origin_colmajor: (n_joints, 16) row-major buffer of flattened column-major 4x4 per joint.
  ChainFK(py::array_t<std::int32_t> joint_types, py::array_t<std::int32_t> q_indices,
          py::array_t<double> T_origin_colmajor, py::array_t<double> axes, std::int32_t n_dof)
      : n_dof_(n_dof) {
    py::buffer_info jt = joint_types.request();
    py::buffer_info qi = q_indices.request();
    py::buffer_info To = T_origin_colmajor.request();
    py::buffer_info ax = axes.request();

    if (jt.ndim != 1 || qi.ndim != 1 || To.ndim != 2 || ax.ndim != 2) {
      throw std::runtime_error("invalid array ranks for ChainFK");
    }
    const ssize_t n = jt.shape[0];
    if (qi.shape[0] != n || To.shape[0] != n || To.shape[1] != 16 || ax.shape[0] != n ||
        ax.shape[1] != 3) {
      throw std::runtime_error("ChainFK: inconsistent joint array shapes");
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

  py::array_t<double> fk_end(py::array_t<double> q) const {
    py::buffer_info qb = q.request();
    if (qb.ndim == 1) {
      if (static_cast<std::int32_t>(qb.shape[0]) != n_dof_) {
        throw std::runtime_error("q length does not match n_dof");
      }
      py::array_t<double> out(std::vector<ssize_t>{1, 4, 4});
      fk_impl(static_cast<const double *>(qb.ptr), 1, static_cast<double *>(out.mutable_data()));
      return out;
    }
    if (qb.ndim != 2) {
      throw std::runtime_error("q must be (n_dof,) or (batch, n_dof)");
    }
    if (static_cast<std::int32_t>(qb.shape[1]) != n_dof_) {
      throw std::runtime_error("q.shape[1] does not match n_dof");
    }
    const ssize_t batch = qb.shape[0];
    py::array_t<double> out(std::vector<ssize_t>{batch, 4, 4});
    fk_impl(static_cast<const double *>(qb.ptr), batch, static_cast<double *>(out.mutable_data()));
    return out;
  }

private:
  void fk_impl(const double *q_ptr, ssize_t batch, double *out_ptr) const {
    for (ssize_t b = 0; b < batch; ++b) {
      const double *qb = q_ptr + b * n_dof_;
      Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
      for (size_t i = 0; i < types_.size(); ++i) {
        double qi = 0.0;
        if (q_idx_[i] >= 0) {
          qi = qb[q_idx_[i]];
        }
        Eigen::Matrix4d Tm;
        switch (types_[i]) {
        case JointType::Fixed:
          Tm = Eigen::Matrix4d::Identity();
          break;
        case JointType::Revolute:
          Tm = rodrigues_motion(axes_[i], qi);
          break;
        case JointType::Prismatic:
          Tm = prismatic_motion(axes_[i], qi);
          break;
        default:
          Tm = Eigen::Matrix4d::Identity();
          break;
        }
        T = T * T_origin_[i] * Tm;
      }
      // Row-major output [b, i, j] matches NumPy default.
      for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
          out_ptr[b * 16 + r * 4 + c] = T(r, c);
        }
      }
    }
  }

  std::int32_t n_dof_;
  std::vector<JointType> types_;
  std::vector<std::int32_t> q_idx_;
  std::vector<Eigen::Matrix4d> T_origin_;
  std::vector<Eigen::Vector3d> axes_;
};

PYBIND11_MODULE(_fk_chain_core, m) {
  m.doc() = "Serial-chain FK (Eigen), RoboCore-compatible with FKSolverNumPy batch path.";
  py::class_<ChainFK>(m, "ChainFK")
      .def(py::init<py::array_t<std::int32_t>, py::array_t<std::int32_t>, py::array_t<double>,
                     py::array_t<double>, std::int32_t>())
      .def("fk_end", &ChainFK::fk_end);
}
