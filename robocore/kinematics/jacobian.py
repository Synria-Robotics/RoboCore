"""Unified Jacobian computation interface.

Provides high-level API for computing geometric Jacobian matrices with
automatic backend selection (NumPy / PyTorch).

Public API:
    jacobian(model, q, *, backend='auto', method='analytic', ...)

Returns 6×n geometric Jacobian matrix (position + orientation derivatives).
"""

from __future__ import annotations
from typing import Any, Sequence, Union

from robocore.kinematics.jacobian_utils.jacobian_solver_numpy import JacobianSolverNumPy

_HAS_TORCH = False
try:  # pragma: no cover
    from robocore.kinematics.jacobian_utils.jacobian_solver_torch import JacobianSolverTorch
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:  # noqa: E722
    torch = None  # type: ignore
    JacobianSolverTorch = None  # type: ignore


def _select_backend(backend: str) -> str:
    """Select backend automatically or validate user choice."""
    if backend == 'auto':
        return 'torch' if _HAS_TORCH else 'numpy'
    if backend not in ('numpy', 'torch'):
        raise ValueError(f"Unsupported backend '{backend}', expected 'auto'|'numpy'|'torch'")
    if backend == 'torch' and not _HAS_TORCH:
        raise RuntimeError("Torch backend requested but PyTorch is not available")
    return backend


def jacobian(
    model: Any,
    q: Sequence[float] | Any,
    *,
    backend: str = 'auto',
    method: str = 'analytic',
    # Numeric Jacobian options
    epsilon: float = 5e-5,
    use_central_diff: bool = True,
    # Torch-specific options
    device: Any | None = None,
    dtype: Any | None = None,
) -> Union[Any, Any]:
    """Compute 6×n geometric Jacobian matrix.

    The Jacobian relates joint velocities to end-effector spatial velocity
    (linear + angular). Uses axis-angle representation for orientation.

    Parameters
    ----------
    model : RobotModel
        Parsed robot model instance.
    q : sequence | tensor
        Joint configuration of length = dof.
    backend : str, default 'auto'
        Backend selection: 'auto' | 'numpy' | 'torch'.
        'auto' prefers torch if available, else numpy.
    method : str, default 'analytic'
        Computation method:
        - 'analytic': Closed-form geometric Jacobian (fastest, most accurate)
        - 'numeric': Finite-difference approximation
        - 'autograd': PyTorch automatic differentiation (torch backend only)
    epsilon : float, default 5e-5
        Finite-difference step size (numeric method only).
    use_central_diff : bool, default True
        Use central differences for numeric method (more accurate than forward).
    device : torch device, optional
        PyTorch device for torch backend (e.g., 'cpu', 'cuda').
    dtype : torch dtype, optional
        PyTorch dtype for torch backend. Defaults to float64 if omitted.

    Returns
    -------
    ndarray | Tensor
        6×n Jacobian matrix. Rows 0-2: linear velocity components (m/s per rad/s),
        Rows 3-5: angular velocity components (rad/s per rad/s).
        Type matches backend: numpy.ndarray or torch.Tensor.

    Raises
    ------
    ValueError
        If backend or method is invalid.
    RuntimeError
        If torch backend requested but unavailable.

    Notes
    -----
    - Analytic method is recommended for production use (fast & accurate).
    - Numeric method useful for validation but slower.
    - Autograd method (torch only) validates against automatic differentiation.
    - Orientation uses axis-angle error representation (not standard geometric).

    Examples
    --------
    >>> J = jacobian(model, q)  # Auto backend, analytic
    >>> J_num = jacobian(model, q, method='numeric')
    >>> J_torch = jacobian(model, q, backend='torch', device='cpu')
    """
    b = _select_backend(backend)
    method = method.lower()

    if b == 'numpy':
        # NumPy backend
        solver = JacobianSolverNumPy(model)
        if method in ('analytic', 'numeric'):
            return solver.solve(q, method=method, epsilon=epsilon, use_central_diff=use_central_diff)
        elif method == 'autograd':
            raise ValueError("Autograd method requires torch backend")
        else:
            raise ValueError(f"Unknown method '{method}' for numpy backend. Use 'analytic' or 'numeric'.")

    # Torch backend
    solver_torch = JacobianSolverTorch(model)  # type: ignore[misc]
    
    # Decide dtype default
    if dtype is None and _HAS_TORCH:  # pragma: no branch
        dtype = torch.float64  # type: ignore[attr-defined]
    
    if method in ('analytic', 'numeric', 'autograd'):
        return solver_torch.solve(
            q,
            method=method,
            epsilon=epsilon,
            use_central_diff=use_central_diff,
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'analytic', 'numeric', or 'autograd'.")


__all__ = ["jacobian"]
