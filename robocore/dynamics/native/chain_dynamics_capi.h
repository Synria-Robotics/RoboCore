// SPDX-License-Identifier: MIT
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ChainDynamicsHandle ChainDynamicsHandle;

ChainDynamicsHandle *chain_dynamics_create(
    const int32_t *parents, const int32_t *joint_types, const int32_t *q_indices,
    const double *T_origin_colmajor, const double *axes, const double *masses,
    const double *coms, const double *inertias_colmajor, const double gravity[3],
    int32_t n_bodies, int32_t n_dof);

void chain_dynamics_destroy(ChainDynamicsHandle *handle);

int32_t chain_dynamics_n_dof(const ChainDynamicsHandle *handle);

void chain_dynamics_rnea(const ChainDynamicsHandle *handle, const double *q, const double *v,
                         const double *a, double *tau_out);

void chain_dynamics_crba(const ChainDynamicsHandle *handle, const double *q, double *M_out);

void chain_dynamics_gravity(const ChainDynamicsHandle *handle, const double *q, double *tau_out);

#ifdef __cplusplus
}
#endif
