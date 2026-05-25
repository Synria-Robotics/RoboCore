# RoboCore Dynamics Benchmark

Fixed-base rigid-body dynamics CPU benchmark. Speedup is normalized to the RoboCore C++/Eigen backend for the same operation and mode.

## Policy

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
| RoboCore | 2.5.0rc4 |
| NumPy | 2.4.6 |
| Pinocchio | 4.0.0 |

## Results

| Operation | Backend | Mode | ms/call | C++ speedup | Error | Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| id | **cpp** | single | 0.006 | 1.0x | 2.22e-16 | ok |
| mass_matrix | **cpp** | single | 0.005 | 1.0x | 3.47e-18 | ok |
| gravity | **cpp** | single | 0.007 | 1.0x | 4.44e-16 | ok |
| nle | **cpp** | single | 0.008 | 1.0x | 1.11e-16 | ok |
| fd | **cpp** | single | 0.009 | 1.0x | 2.09e-11 | ok |
| id | **cpp** | batch | 0.443 | 1.0x | - | ok |
| mass_matrix | **cpp** | batch | 0.436 | 1.0x | - | ok |
| fd | **cpp** | batch | 0.963 | 1.0x | - | ok |
| id | numpy | single | 0.862 | 141.1x | 0.00e+00 | ok |
| mass_matrix | numpy | single | 0.881 | 174.9x | 0.00e+00 | ok |
| gravity | numpy | single | 0.863 | 115.6x | 0.00e+00 | ok |
| nle | numpy | single | 0.862 | 111.2x | 0.00e+00 | ok |
| fd | numpy | single | 1.758 | 197.3x | 0.00e+00 | ok |
| id | pinocchio | single | 0.036 | 5.8x | 4.44e-16 | ok |
| mass_matrix | pinocchio | single | 0.036 | 7.1x | 1.39e-17 | ok |
| gravity | pinocchio | single | 0.036 | 4.8x | 4.44e-16 | ok |
| nle | pinocchio | single | 0.036 | 4.6x | 7.77e-16 | ok |
| fd | pinocchio | single | 0.036 | 4.0x | 7.28e-12 | ok |
