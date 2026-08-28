// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

namespace robocore_dynamics {

enum class JointType : std::int32_t { Fixed = 0, Revolute = 1, Prismatic = 2 };

using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat6 = Eigen::Matrix<double, 6, 6>;

class ChainDynamics {
public:
  ChainDynamics(const std::int32_t *parents, const std::int32_t *joint_types,
                const std::int32_t *q_indices, const double *T_origin_colmajor,
                const double *axes, const double *masses, const double *coms,
                const double *inertias_colmajor, const double gravity[3],
                std::int32_t n_bodies, std::int32_t n_dof);

  std::int32_t n_dof() const { return n_dof_; }

  void rnea(const double *q, const double *v, const double *a, double *tau_out) const;

  /// Row-major ``n_dof x n_dof`` mass matrix.
  void crba(const double *q, double *M_out) const;

  void gravity(const double *q, double *tau_out) const {
    std::vector<double> zeros(static_cast<size_t>(n_dof_), 0.0);
    rnea(q, zeros.data(), zeros.data(), tau_out);
  }

private:
  void build_kinematics(const double *q, std::vector<Mat6> &Xup, std::vector<Vec6> &S,
                        std::vector<Mat6> &I) const;

  Eigen::VectorXd rnea_one(const double *q, const double *v, const double *a) const;

  Eigen::MatrixXd crba_one(const double *q) const;

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

}  // namespace robocore_dynamics
