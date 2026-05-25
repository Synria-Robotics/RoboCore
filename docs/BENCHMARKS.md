# RoboCore Benchmarks

This document records reproducible benchmark runs used for release planning.
Numbers are machine dependent; use the JSON files under `benchmark-results/`
as the source of truth for policy metadata and raw values.

## Dynamics CPU Benchmark

This run is the v2.6.0 dynamics development baseline. It validates fixed-base
rigid-body dynamics on the Alicia-D 6-DOF arm chain and compares RoboCore
C++/Eigen against RoboCore NumPy and Pinocchio.

Command:

```bash
python -m robocore.benchmark dynamics \
  --backends cpp,numpy,pinocchio \
  --samples 200 \
  --repeats 5 \
  --inner-single 1000 \
  --inner-batch 200 \
  --output-dir benchmark-results/v2.6.0-dynamics-cpu
```

Policy:

| Field | Value |
| --- | --- |
| Python | 3.11.13 |
| Platform | Linux-5.15.0-139-generic-x86_64-with-glibc2.31 |
| Processor | x86_64 |
| Robot | alicia_d |
| Chain | base_link -> tool0 (6 DOF) |
| Samples | 200 |
| Seed | 0 |
| Repeats | 5 |
| Inner single loops | 1000 |
| Inner batch loops | 200 |
| RoboCore | 2.5.0rc4 development tree |
| NumPy | 2.4.6 |
| Pinocchio | 4.0.0 |

Single-call CPU results. `C++ advantage` is `other_time / robocore_cpp_time`.

| Operation | **RoboCore C++** (ms) | NumPy (ms) | C++ vs NumPy | Pinocchio (ms) | C++ vs Pinocchio | Max error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RNEA inverse dynamics | **0.006** | 0.862 | 141.1x | 0.036 | 5.8x | 4.44e-16 |
| CRBA mass matrix | **0.005** | 0.881 | 174.9x | 0.036 | 7.1x | 1.39e-17 |
| Gravity | **0.007** | 0.863 | 115.6x | 0.036 | 4.8x | 4.44e-16 |
| Nonlinear effects | **0.008** | 0.862 | 111.2x | 0.036 | 4.6x | 7.77e-16 |
| Forward dynamics solve | **0.009** | 1.758 | 197.3x | 0.036 | 4.0x | 7.28e-12 |

C++ batch path results for 200 samples per call:

| Operation | C++ batch ms/call | Status |
| --- | ---: | --- |
| RNEA inverse dynamics | 0.443 | ok |
| CRBA mass matrix | 0.436 | ok |
| Forward dynamics solve | 0.963 | ok |

Interpretation:

- RoboCore C++/Eigen is the default recommended backend for low-latency
  fixed-base dynamics.
- Pinocchio remains the external correctness and capability reference. This
  benchmark compares the same projected 6-DOF chain, not the full URDF tree.
- Current forward dynamics uses CRBA plus nonlinear effects and a linear solve:
  `M(q) ddq = tau - nle(q, v)`. True O(n) ABA is a v2.6+ implementation target.

Artifacts:

- `benchmark-results/v2.6.0-dynamics-cpu/dynamics.json`
- `benchmark-results/v2.6.0-dynamics-cpu/dynamics.md`
