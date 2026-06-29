# RootSolvers.jl Release Notes

main
-------

v1.0.3
-------
- Improved robustness of Newton's method by implementing an inline finite-difference fallback for singular derivatives. This avoids the loss of iteration history and improves performance by keeping the kernel a single iteration loop.
- Replaced short-circuiting branches (`||`) with bitwise operators (`|`) in tolerance checks and solver loops to improve GPU performance and reduce warp divergence.
