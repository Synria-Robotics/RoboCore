// SPDX-License-Identifier: MIT

#include "chain_dynamics_capi.h"

#include <new>

#include "chain_dynamics.hpp"

struct ChainDynamicsHandle {
  robocore_dynamics::ChainDynamics inner;
};

ChainDynamicsHandle *chain_dynamics_create(
    const int32_t *parents, const int32_t *joint_types, const int32_t *q_indices,
    const double *T_origin_colmajor, const double *axes, const double *masses,
    const double *coms, const double *inertias_colmajor, const double gravity[3],
    const int32_t n_bodies, const int32_t n_dof) {
  try {
    auto *handle = new ChainDynamicsHandle{
        robocore_dynamics::ChainDynamics(parents, joint_types, q_indices, T_origin_colmajor, axes,
                                         masses, coms, inertias_colmajor, gravity, n_bodies,
                                         n_dof)};
    return handle;
  } catch (...) {
    return nullptr;
  }
}

void chain_dynamics_destroy(ChainDynamicsHandle *handle) { delete handle; }

int32_t chain_dynamics_n_dof(const ChainDynamicsHandle *handle) {
  return handle ? handle->inner.n_dof() : 0;
}

void chain_dynamics_rnea(const ChainDynamicsHandle *handle, const double *q, const double *v,
                         const double *a, double *tau_out) {
  if (handle == nullptr) {
    return;
  }
  handle->inner.rnea(q, v, a, tau_out);
}

void chain_dynamics_crba(const ChainDynamicsHandle *handle, const double *q, double *M_out) {
  if (handle == nullptr) {
    return;
  }
  handle->inner.crba(q, M_out);
}

void chain_dynamics_gravity(const ChainDynamicsHandle *handle, const double *q, double *tau_out) {
  if (handle == nullptr) {
    return;
  }
  handle->inner.gravity(q, tau_out);
}
