// SPDX-License-Identifier: MIT
// pybind11 wrapper around native fixed-base chain dynamics (RNEA, CRBA, FD).

#include <cstdint>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

#include "native/chain_dynamics.hpp"

namespace py = pybind11;
using robocore_dynamics::ChainDynamics;

class ChainDynamicsPy {
public:
  ChainDynamicsPy(py::array_t<std::int32_t> parents, py::array_t<std::int32_t> joint_types,
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

    const auto *pp = static_cast<const std::int32_t *>(pb.ptr);
    const auto *tp = static_cast<const std::int32_t *>(tb.ptr);
    const auto *qp = static_cast<const std::int32_t *>(qb.ptr);
    const auto *op = static_cast<const double *>(ob.ptr);
    const auto *ap = static_cast<const double *>(ab.ptr);
    const auto *mp = static_cast<const double *>(mb.ptr);
    const auto *cp = static_cast<const double *>(cb.ptr);
    const auto *ip = static_cast<const double *>(ib.ptr);
    const auto *gp = static_cast<const double *>(gb.ptr);
    const double grav[3] = {gp[0], gp[1], gp[2]};

    core_ = std::make_unique<ChainDynamics>(
        pp, tp, qp, op, ap, mp, cp, ip, grav, static_cast<std::int32_t>(n), n_dof_);
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
    std::vector<double> tau(static_cast<size_t>(n_dof_));
    for (py::ssize_t b = 0; b < Q.batch; ++b) {
      core_->rnea(Q.ptr + b * n_dof_, V.ptr + b * n_dof_, A.ptr + b * n_dof_, tau.data());
      for (int j = 0; j < n_dof_; ++j) {
        outp[b * n_dof_ + j] = tau[static_cast<size_t>(j)];
      }
    }
    return out;
  }

  py::array_t<double> crba(py::array_t<double> q) const {
    const auto Q = normalize_qva(q, "q");
    py::array_t<double> out({Q.batch, static_cast<py::ssize_t>(n_dof_),
                             static_cast<py::ssize_t>(n_dof_)});
    double *outp = static_cast<double *>(out.mutable_data());
    const size_t mm = static_cast<size_t>(n_dof_) * static_cast<size_t>(n_dof_);
    std::vector<double> M(mm);
    for (py::ssize_t b = 0; b < Q.batch; ++b) {
      core_->crba(Q.ptr + b * n_dof_, M.data());
      for (size_t k = 0; k < mm; ++k) {
        outp[b * static_cast<py::ssize_t>(mm) + static_cast<py::ssize_t>(k)] = M[k];
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
    const size_t mm = static_cast<size_t>(n_dof_) * static_cast<size_t>(n_dof_);
    std::vector<double> nle(static_cast<size_t>(n_dof_));
    std::vector<double> M(mm);
    std::vector<double> zeros(static_cast<size_t>(n_dof_), 0.0);
    for (py::ssize_t b = 0; b < Q.batch; ++b) {
      const double *qrow = Q.ptr + b * n_dof_;
      const double *vrow = V.ptr + b * n_dof_;
      const double *trow = T.ptr + b * n_dof_;
      core_->rnea(qrow, vrow, zeros.data(), nle.data());
      core_->crba(qrow, M.data());
      Eigen::Map<const Eigen::MatrixXd> Mmap(M.data(), n_dof_, n_dof_);
      Eigen::VectorXd rhs(n_dof_);
      for (int j = 0; j < n_dof_; ++j) {
        rhs[j] = trow[j] - nle[static_cast<size_t>(j)];
      }
      const Eigen::VectorXd ddq = Mmap.ldlt().solve(rhs);
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

  std::int32_t n_dof_;
  std::unique_ptr<ChainDynamics> core_;
};

PYBIND11_MODULE(_dynamics_core, m) {
  py::class_<ChainDynamicsPy>(m, "ChainDynamics")
      .def(py::init<py::array_t<std::int32_t>, py::array_t<std::int32_t>,
                    py::array_t<std::int32_t>, py::array_t<double>, py::array_t<double>,
                    py::array_t<double>, py::array_t<double>, py::array_t<double>,
                    py::array_t<double>, std::int32_t>())
      .def("rnea", &ChainDynamicsPy::rnea)
      .def("crba", &ChainDynamicsPy::crba)
      .def("forward_dynamics", &ChainDynamicsPy::forward_dynamics);
}
