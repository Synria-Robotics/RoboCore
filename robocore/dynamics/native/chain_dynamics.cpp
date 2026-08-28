// SPDX-License-Identifier: MIT

#include "chain_dynamics.hpp"

namespace robocore_dynamics {

namespace {

using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat6 = Eigen::Matrix<double, 6, 6>;

Eigen::Matrix3d skew(const Eigen::Vector3d &v) {
  Eigen::Matrix3d S;
  S << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return S;
}

Eigen::Matrix4d rodrigues_motion(const Eigen::Vector3d &axis_raw, double angle) {
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

Eigen::Matrix4d prismatic_motion(const Eigen::Vector3d &axis_raw, double q) {
  Eigen::Vector3d a = axis_raw;
  const double n = a.norm();
  if (n > 1e-10) {
    a /= n;
  }
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 1>(0, 3) = a * q;
  return T;
}

Mat6 plucker_x(const Eigen::Matrix4d &T_parent_child) {
  const Eigen::Matrix3d R = T_parent_child.block<3, 3>(0, 0);
  const Eigen::Vector3d p = T_parent_child.block<3, 1>(0, 3);
  const Eigen::Matrix3d Rt = R.transpose();
  Mat6 X = Mat6::Zero();
  X.block<3, 3>(0, 0) = Rt;
  X.block<3, 3>(3, 0) = -Rt * skew(p);
  X.block<3, 3>(3, 3) = Rt;
  return X;
}

Mat6 ad_vec(const Vec6 &v) {
  const Eigen::Vector3d w = v.segment<3>(0);
  const Eigen::Vector3d u = v.segment<3>(3);
  Mat6 A = Mat6::Zero();
  A.block<3, 3>(0, 0) = skew(w);
  A.block<3, 3>(3, 0) = skew(u);
  A.block<3, 3>(3, 3) = skew(w);
  return A;
}

Vec6 motion_subspace(JointType type, const Eigen::Vector3d &axis_raw) {
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

Mat6 spatial_inertia(double mass, const Eigen::Vector3d &com, const Eigen::Matrix3d &I_com) {
  Mat6 I = Mat6::Zero();
  const Eigen::Matrix3d I_origin =
      I_com + mass * (com.dot(com) * Eigen::Matrix3d::Identity() - com * com.transpose());
  I.block<3, 3>(0, 0) = I_origin;
  I.block<3, 3>(0, 3) = skew(mass * com);
  I.block<3, 3>(3, 0) = skew(mass * com).transpose();
  I.block<3, 3>(3, 3) = mass * Eigen::Matrix3d::Identity();
  return I;
}

}  // namespace

ChainDynamics::ChainDynamics(const std::int32_t *parents, const std::int32_t *joint_types,
                             const std::int32_t *q_indices, const double *T_origin_colmajor,
                             const double *axes, const double *masses, const double *coms,
                             const double *inertias_colmajor, const double gravity[3],
                             const std::int32_t n_bodies, const std::int32_t n_dof)
    : n_dof_(n_dof) {
  if (n_bodies <= 0 || n_dof <= 0) {
    throw std::runtime_error("ChainDynamics: n_bodies and n_dof must be positive");
  }

  parents_.resize(static_cast<size_t>(n_bodies));
  types_.resize(static_cast<size_t>(n_bodies));
  q_idx_.resize(static_cast<size_t>(n_bodies));
  T_origin_.resize(static_cast<size_t>(n_bodies));
  axes_.resize(static_cast<size_t>(n_bodies));
  masses_.resize(static_cast<size_t>(n_bodies));
  coms_.resize(static_cast<size_t>(n_bodies));
  inertias_.resize(static_cast<size_t>(n_bodies));

  for (std::int32_t i = 0; i < n_bodies; ++i) {
    const size_t k = static_cast<size_t>(i);
    parents_[k] = parents[i];
    types_[k] = static_cast<JointType>(joint_types[i]);
    q_idx_[k] = q_indices[i];
    Eigen::Map<const Eigen::Matrix4d> T(T_origin_colmajor + i * 16);
    T_origin_[k] = T;
    axes_[k] = Eigen::Vector3d(axes[i * 3 + 0], axes[i * 3 + 1], axes[i * 3 + 2]);
    masses_[k] = masses[i];
    coms_[k] = Eigen::Vector3d(coms[i * 3 + 0], coms[i * 3 + 1], coms[i * 3 + 2]);
    Eigen::Matrix3d Ic;
    Ic << inertias_colmajor[i * 9 + 0], inertias_colmajor[i * 9 + 1],
        inertias_colmajor[i * 9 + 2], inertias_colmajor[i * 9 + 3],
        inertias_colmajor[i * 9 + 4], inertias_colmajor[i * 9 + 5],
        inertias_colmajor[i * 9 + 6], inertias_colmajor[i * 9 + 7],
        inertias_colmajor[i * 9 + 8];
    inertias_[k] = Ic;
  }
  gravity_ = Eigen::Vector3d(gravity[0], gravity[1], gravity[2]);
}

void ChainDynamics::rnea(const double *q, const double *v, const double *a,
                         double *tau_out) const {
  const Eigen::VectorXd tau = rnea_one(q, v, a);
  for (int j = 0; j < n_dof_; ++j) {
    tau_out[j] = tau[j];
  }
}

void ChainDynamics::crba(const double *q, double *M_out) const {
  const Eigen::MatrixXd M = crba_one(q);
  for (int r = 0; r < n_dof_; ++r) {
    for (int c = 0; c < n_dof_; ++c) {
      M_out[r * n_dof_ + c] = M(r, c);
    }
  }
}

void ChainDynamics::build_kinematics(const double *q, std::vector<Mat6> &Xup, std::vector<Vec6> &S,
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

Eigen::VectorXd ChainDynamics::rnea_one(const double *q, const double *v, const double *a) const {
  std::vector<Mat6> Xup;
  std::vector<Vec6> S;
  std::vector<Mat6> I;
  build_kinematics(q, Xup, S, I);
  const size_t nb = parents_.size();
  std::vector<Vec6> V(nb, Vec6::Zero());
  std::vector<Vec6> A_vec(nb, Vec6::Zero());
  std::vector<Vec6> F(nb, Vec6::Zero());
  Eigen::VectorXd tau = Eigen::VectorXd::Zero(n_dof_);

  Vec6 a0 = Vec6::Zero();
  a0.segment<3>(3) = -gravity_;

  for (size_t i = 0; i < nb; ++i) {
    const int parent = parents_[i];
    if (parent < 0) {
      V[i].setZero();
      A_vec[i] = Xup[i] * a0;
    } else {
      const int qi = q_idx_[i];
      const double qd = qi >= 0 ? v[qi] : 0.0;
      const double qdd = qi >= 0 ? a[qi] : 0.0;
      V[i] = Xup[i] * V[static_cast<size_t>(parent)] + S[i] * qd;
      A_vec[i] = Xup[i] * A_vec[static_cast<size_t>(parent)] + S[i] * qdd +
                 ad_vec(V[i]) * (S[i] * qd);
    }
  }

  for (size_t i = 0; i < nb; ++i) {
    F[i] = I[i] * A_vec[i] - ad_vec(V[i]).transpose() * (I[i] * V[i]);
  }
  for (int i = static_cast<int>(nb) - 1; i >= 0; --i) {
    const int parent = parents_[static_cast<size_t>(i)];
    if (parent >= 0) {
      F[static_cast<size_t>(parent)] +=
          Xup[static_cast<size_t>(i)].transpose() * F[static_cast<size_t>(i)];
    }
    const int qi = q_idx_[static_cast<size_t>(i)];
    if (qi >= 0) {
      tau[qi] = S[static_cast<size_t>(i)].dot(F[static_cast<size_t>(i)]);
    }
  }
  return tau;
}

Eigen::MatrixXd ChainDynamics::crba_one(const double *q) const {
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
          Xup[static_cast<size_t>(j)].transpose() * Ic[static_cast<size_t>(j)] *
          Xup[static_cast<size_t>(j)];
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

}  // namespace robocore_dynamics
