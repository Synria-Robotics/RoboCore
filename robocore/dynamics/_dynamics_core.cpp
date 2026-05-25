// SPDX-License-Identifier: MIT
// Fixed-base chain dynamics (RNEA, CRBA, forward solve), Eigen + pybind11.

#include <Eigen/Dense>
#include <cstdint>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

enum class JointType : std::int32_t { Fixed = 0, Revolute = 1, Prismatic = 2 };

using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat6 = Eigen::Matrix<double, 6, 6>;

static Eigen::Matrix3d skew(const Eigen::Vector3d &v) {
  Eigen::Matrix3d S;
  S << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return S;
}

static Eigen::Matrix4d rodrigues_motion(const Eigen::Vector3d &axis_raw, double angle) {
  Eigen::Vector3d k = axis_raw;
  const double n = k.norm();
  if (n < 1e-10) {
    return Eigen::Matrix4d::Identity();
  }
  k /= n;
  const double x = k[0], y = k[1], z = k[2];
  Eigen::Matrix3d K;
  K << 0.0, -z, y, z, 0.0, -x, -y, x, 0.0;
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() + s * K + (1.0 - c) * K * K;
  return T;
}

static Eigen::Matrix4d prismatic_motion(const Eigen::Vector3d &axis_raw, double q) {
  Eigen::Vector3d a = axis_raw;
  const double n = a.norm();
  if (n > 1e-10) {
    a /= n;
  }
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 1>(0, 3) = a * q;
  return T;
}

static Mat6 plucker_x(const Eigen::Matrix4d &T_parent_child) {
  const Eigen::Matrix3d R = T_parent_child.block<3, 3>(0, 0);
  const Eigen::Vector3d p = T_parent_child.block<3, 1>(0, 3);
  const Eigen::Matrix3d Rt = R.transpose();
  Mat6 X = Mat6::Zero();
  X.block<3, 3>(0, 0) = Rt;
  X.block<3, 3>(3, 0) = -Rt * skew(p);
  X.block<3, 3>(3, 3) = Rt;
  return X;
}

static Mat6 ad_vec(const Vec6 &v) {
  const Eigen::Vector3d w = v.segment<3>(0);
  const Eigen::Vector3d u = v.segment<3>(3);
  Mat6 A = Mat6::Zero();
  A.block<3, 3>(0, 0) = skew(w);
  A.block<3, 3>(3, 0) = skew(u);
  A.block<3, 3>(3, 3) = skew(w);
  return A;
}

static Vec6 motion_subspace(JointType type, const Eigen::Vector3d &axis_raw) {
  Eigen::Vector3d a = axis_raw;
  const double n = a.norm();
  if (n < 1e-10) {
    a = Eigen::Vector3d(0.0, 0.0, 1.0);
  } else {
    a /= n;
  }
  Vec6 S = Vec6::Zero();
  if (type == JointType::Revolute) {
    S.segment<3>(0) = a;
  } else if (type == JointType::Prismatic) {
    S.segment<3>(3) = a;
  }
  return S;
}

static Mat6 spatial_inertia(double mass, const Eigen::Vector3d &com,
                            const Eigen::Matrix3d &I_com) {
  Mat6 I = Mat6::Zero();
  const Eigen::Matrix3d C = skew(com);
  const Eigen::Matrix3d I_origin =
      I_com + mass * (com.dot(com) * Eigen::Matrix3d::Identity() - com * com.transpose());
  I.block<3, 3>(0, 0) = I_origin;
  I.block<3, 3>(0, 3) = skew(mass * com);
  I.block<3, 3>(3, 0) = skew(mass * com).transpose();
  I.block<3, 3>(3, 3) = mass * Eigen::Matrix3d::Identity();
  (void)C;
  return I;
}

class ChainDynamics {
public:
  ChainDynamics(py::array_t<std::int32_t> parents, py::array_t<std::int32_t> joint_types,
                py::array_t<std::int32_t> q_indices, py::array_t<double> T_origin_colmajor,
                py::array_t<double> axes, py::array_t<double> masses, py::array_t<double> coms,
                py::array_t<double> inertias, py::array_t<double> gravity, std::int32_t n_dof)
      : n_dof_(n_dof) {
    const py::buffer_info pb = parents.request();
    const py::buffer_info tb = joint_types.request();
    const py::buffer_info qb = q_indices.request();
    const py::buffer_info ob = T_origin_colmajor.request();
    const py::buffer_info ab = axes.request();
    const py::buffer_info mb = masses.request();
    const py::buffer_info cb = coms.request();
    const py::buffer_info ib = inertias.request();
    const py::buffer_info gb = gravity.request();

    if (pb.ndim != 1 || tb.ndim != 1 || qb.ndim != 1 || ob.ndim != 2 || ab.ndim != 2 ||
        mb.ndim != 1 || cb.ndim != 2 || ib.ndim != 2 || gb.ndim != 1) {
      throw std::runtime_error("ChainDynamics: invalid array ranks");
    }
    const py::ssize_t n = pb.shape[0];
    if (tb.shape[0] != n || qb.shape[0] != n || ob.shape[0] != n || ob.shape[1] != 16 ||
        ab.shape[0] != n || ab.shape[1] != 3 || mb.shape[0] != n || cb.shape[0] != n ||
        cb.shape[1] != 3 || ib.shape[0] != n || ib.shape[1] != 9 || gb.shape[0] != 3) {
      throw std::runtime_error("ChainDynamics: inconsistent array shapes");
    }

    parents_.resize(static_cast<size_t>(n));
    types_.resize(static_cast<size_t>(n));
    q_idx_.resize(static_cast<size_t>(n));
    T_origin_.resize(static_cast<size_t>(n));
    axes_.resize(static_cast<size_t>(n));
    masses_.resize(static_cast<size_t>(n));
    coms_.resize(static_cast<size_t>(n));
    inertias_.resize(static_cast<size_t>(n));

    const auto *pp = static_cast<const std::int32_t *>(pb.ptr);
    const auto *tp = static_cast<const std::int32_t *>(tb.ptr);
    const auto *qp = static_cast<const std::int32_t *>(qb.ptr);
    const auto *op = static_cast<const double *>(ob.ptr);
    const auto *ap = static_cast<const double *>(ab.ptr);
    const auto *mp = static_cast<const double *>(mb.ptr);
    const auto *cp = static_cast<const double *>(cb.ptr);
    const auto *ip = static_cast<const double *>(ib.ptr);
    const auto *gp = static_cast<const double *>(gb.ptr);

    for (py::ssize_t i = 0; i < n; ++i) {
      const size_t k = static_cast<size_t>(i);
      parents_[k] = pp[i];
      types_[k] = static_cast<JointType>(tp[i]);
      q_idx_[k] = qp[i];
      Eigen::Map<const Eigen::Matrix4d> T(op + i * 16);
      T_origin_[k] = T;
      axes_[k] = Eigen::Vector3d(ap[i * 3 + 0], ap[i * 3 + 1], ap[i * 3 + 2]);
      masses_[k] = mp[i];
      coms_[k] = Eigen::Vector3d(cp[i * 3 + 0], cp[i * 3 + 1], cp[i * 3 + 2]);
      Eigen::Matrix3d Ic;
      Ic << ip[i * 9 + 0], ip[i * 9 + 1], ip[i * 9 + 2], ip[i * 9 + 3], ip[i * 9 + 4],
          ip[i * 9 + 5], ip[i * 9 + 6], ip[i * 9 + 7], ip[i * 9 + 8];
      inertias_[k] = Ic;
    }
    gravity_ = Eigen::Vector3d(gp[0], gp[1], gp[2]);
  }

  py::array_t<double> rnea(py::array_t<double> q, py::array_t<double> v,
                           py::array_t<double> a) const {
    const auto Q = normalize_qva(q, "q");
    const auto V = normalize_qva(v, "v");
    const auto A = normalize_qva(a, "a");
    if (V.batch != Q.batch || A.batch != Q.batch) {
      throw std::runtime_error("rnea batch size mismatch");
    }
    py::array_t<double> out({Q.batch, static_cast<py::ssize_t>(n_dof_)});
    double *outp = static_cast<double *>(out.mutable_data());
    for (py::ssize_t b = 0; b < Q.batch; ++b) {
      const Eigen::VectorXd tau =
          rnea_one(Q.ptr + b * n_dof_, V.ptr + b * n_dof_, A.ptr + b * n_dof_);
      for (int j = 0; j < n_dof_; ++j) {
        outp[b * n_dof_ + j] = tau[j];
      }
    }
    return out;
  }

  py::array_t<double> crba(py::array_t<double> q) const {
    const auto Q = normalize_qva(q, "q");
    py::array_t<double> out({Q.batch, static_cast<py::ssize_t>(n_dof_),
                             static_cast<py::ssize_t>(n_dof_)});
    double *outp = static_cast<double *>(out.mutable_data());
    for (py::ssize_t b = 0; b < Q.batch; ++b) {
      const Eigen::MatrixXd M = crba_one(Q.ptr + b * n_dof_);
      for (int r = 0; r < n_dof_; ++r) {
        for (int c = 0; c < n_dof_; ++c) {
          outp[b * n_dof_ * n_dof_ + r * n_dof_ + c] = M(r, c);
        }
      }
    }
    return out;
  }

  py::array_t<double> forward_dynamics(py::array_t<double> q, py::array_t<double> v,
                                       py::array_t<double> tau) const {
    const auto Q = normalize_qva(q, "q");
    const auto V = normalize_qva(v, "v");
    const auto T = normalize_qva(tau, "tau");
    if (V.batch != Q.batch || T.batch != Q.batch) {
      throw std::runtime_error("forward_dynamics batch size mismatch");
    }
    py::array_t<double> out({Q.batch, static_cast<py::ssize_t>(n_dof_)});
    double *outp = static_cast<double *>(out.mutable_data());
    std::vector<double> zeros(static_cast<size_t>(n_dof_), 0.0);
    for (py::ssize_t b = 0; b < Q.batch; ++b) {
      const double *qrow = Q.ptr + b * n_dof_;
      const double *vrow = V.ptr + b * n_dof_;
      const double *trow = T.ptr + b * n_dof_;
      const Eigen::VectorXd nle = rnea_one(qrow, vrow, zeros.data());
      const Eigen::MatrixXd M = crba_one(qrow);
      Eigen::VectorXd rhs(n_dof_);
      for (int j = 0; j < n_dof_; ++j) {
        rhs[j] = trow[j] - nle[j];
      }
      const Eigen::VectorXd ddq = M.ldlt().solve(rhs);
      for (int j = 0; j < n_dof_; ++j) {
        outp[b * n_dof_ + j] = ddq[j];
      }
    }
    return out;
  }

private:
  struct BatchView {
    const double *ptr;
    py::ssize_t batch;
  };

  BatchView normalize_qva(py::array_t<double> x, const char *name) const {
    const py::buffer_info xb = x.request();
    if (xb.ndim == 1) {
      if (static_cast<std::int32_t>(xb.shape[0]) != n_dof_) {
        throw std::runtime_error(std::string(name) + " length does not match n_dof");
      }
      return {static_cast<const double *>(xb.ptr), 1};
    }
    if (xb.ndim == 2) {
      if (static_cast<std::int32_t>(xb.shape[1]) != n_dof_) {
        throw std::runtime_error(std::string(name) + ".shape[1] does not match n_dof");
      }
      return {static_cast<const double *>(xb.ptr), xb.shape[0]};
    }
    throw std::runtime_error(std::string(name) + " must be (n_dof,) or (batch,n_dof)");
  }

  void build_kinematics(const double *q, std::vector<Mat6> &Xup, std::vector<Vec6> &S,
                        std::vector<Mat6> &I) const {
    const size_t nb = parents_.size();
    Xup.resize(nb);
    S.resize(nb);
    I.resize(nb);
    for (size_t i = 0; i < nb; ++i) {
      double qi = 0.0;
      if (q_idx_[i] >= 0) {
        qi = q[q_idx_[i]];
      }
      Eigen::Matrix4d Tm = Eigen::Matrix4d::Identity();
      if (types_[i] == JointType::Revolute) {
        Tm = rodrigues_motion(axes_[i], qi);
      } else if (types_[i] == JointType::Prismatic) {
        Tm = prismatic_motion(axes_[i], qi);
      }
      const Eigen::Matrix4d Tpc = T_origin_[i] * Tm;
      Xup[i] = plucker_x(Tpc);
      S[i] = motion_subspace(types_[i], axes_[i]);
      I[i] = spatial_inertia(masses_[i], coms_[i], inertias_[i]);
    }
  }

  Eigen::VectorXd rnea_one(const double *q, const double *v, const double *a) const {
    std::vector<Mat6> Xup;
    std::vector<Vec6> S;
    std::vector<Mat6> I;
    build_kinematics(q, Xup, S, I);
    const size_t nb = parents_.size();
    std::vector<Vec6> V(nb, Vec6::Zero());
    std::vector<Vec6> A(nb, Vec6::Zero());
    std::vector<Vec6> F(nb, Vec6::Zero());
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(n_dof_);

    Vec6 a0 = Vec6::Zero();
    a0.segment<3>(3) = -gravity_;

    for (size_t i = 0; i < nb; ++i) {
      const int parent = parents_[i];
      if (parent < 0) {
        V[i].setZero();
        A[i] = Xup[i] * a0;
      } else {
        const int qi = q_idx_[i];
        const double qd = qi >= 0 ? v[qi] : 0.0;
        const double qdd = qi >= 0 ? a[qi] : 0.0;
        V[i] = Xup[i] * V[static_cast<size_t>(parent)] + S[i] * qd;
        A[i] = Xup[i] * A[static_cast<size_t>(parent)] + S[i] * qdd +
               ad_vec(V[i]) * (S[i] * qd);
      }
    }

    for (size_t i = 0; i < nb; ++i) {
      F[i] = I[i] * A[i] - ad_vec(V[i]).transpose() * (I[i] * V[i]);
    }
    for (int i = static_cast<int>(nb) - 1; i >= 0; --i) {
      const int parent = parents_[static_cast<size_t>(i)];
      if (parent >= 0) {
        F[static_cast<size_t>(parent)] += Xup[static_cast<size_t>(i)].transpose() * F[static_cast<size_t>(i)];
      }
      const int qi = q_idx_[static_cast<size_t>(i)];
      if (qi >= 0) {
        tau[qi] = S[static_cast<size_t>(i)].dot(F[static_cast<size_t>(i)]);
      }
    }
    return tau;
  }

  Eigen::MatrixXd crba_one(const double *q) const {
    std::vector<Mat6> Xup;
    std::vector<Vec6> S;
    std::vector<Mat6> I;
    build_kinematics(q, Xup, S, I);
    const size_t nb = parents_.size();
    std::vector<Mat6> Ic = I;

    for (int j = static_cast<int>(nb) - 1; j > 0; --j) {
      const int parent = parents_[static_cast<size_t>(j)];
      if (parent >= 0) {
        Ic[static_cast<size_t>(parent)] +=
            Xup[static_cast<size_t>(j)].transpose() * Ic[static_cast<size_t>(j)] * Xup[static_cast<size_t>(j)];
      }
    }

    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n_dof_, n_dof_);
    for (size_t j = 0; j < nb; ++j) {
      const int qj = q_idx_[j];
      if (qj < 0) {
        continue;
      }
      int from = static_cast<int>(j);
      Vec6 F = Ic[j] * S[j];
      M(qj, qj) = S[j].dot(F);
      while (parents_[static_cast<size_t>(from)] >= 0) {
        const int to = parents_[static_cast<size_t>(from)];
        F = Xup[static_cast<size_t>(from)].transpose() * F;
        const int qi = q_idx_[static_cast<size_t>(to)];
        if (qi >= 0) {
          M(qi, qj) = S[static_cast<size_t>(to)].dot(F);
        }
        from = to;
      }
    }
    const Eigen::VectorXd diag = M.diagonal();
    M = (M + M.transpose()).eval();
    M.diagonal() = diag;
    return M;
  }

  std::int32_t n_dof_;
  std::vector<std::int32_t> parents_;
  std::vector<JointType> types_;
  std::vector<std::int32_t> q_idx_;
  std::vector<Eigen::Matrix4d> T_origin_;
  std::vector<Eigen::Vector3d> axes_;
  std::vector<double> masses_;
  std::vector<Eigen::Vector3d> coms_;
  std::vector<Eigen::Matrix3d> inertias_;
  Eigen::Vector3d gravity_;
};

PYBIND11_MODULE(_dynamics_core, m) {
  py::class_<ChainDynamics>(m, "ChainDynamics")
      .def(py::init<py::array_t<std::int32_t>, py::array_t<std::int32_t>,
                    py::array_t<std::int32_t>, py::array_t<double>, py::array_t<double>,
                    py::array_t<double>, py::array_t<double>, py::array_t<double>,
                    py::array_t<double>, std::int32_t>())
      .def("rnea", &ChainDynamics::rnea)
      .def("crba", &ChainDynamics::crba)
      .def("forward_dynamics", &ChainDynamics::forward_dynamics);
}
