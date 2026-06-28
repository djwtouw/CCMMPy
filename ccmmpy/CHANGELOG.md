# Changelog

## ccmmpy 0.2.2

+ Fixed an integer overflow in the computation of the mean squared distance
  used to scale the weights in `SparseWeights`. For very large numbers of
  observations this could produce incorrect weights (in some cases larger
  than one).

+ Fixed an off-by-one error in the median used to determine the default
  fusion threshold, which could return a slightly incorrect value when the
  number of pairwise distances was even.

+ Fixed `SparseWeights.to_dense()`, which applied the Gaussian kernel a second
  time and therefore returned values that did not match `values()`.

+ Fixed `SparseWeights` with `connection_type="MST"`, which raised an error on
  NumPy >= 2.0 when assigning a between-cluster distance.

+ Switched the build in `setup.py` from the removed `distutils` to
  `setuptools`, so the extension builds on Python >= 3.12.

+ Added regression tests that ship with the package and run against an
  installed copy with `pytest --pyargs ccmmpy.tests`. These use the same
  `two_half_moons` data set as the R package and pin the same reference values.


## ccmmpy 0.2

+ Added a new option to guarantee a connected weight matrix in `SparseWeights`,
  selected through the `connection_type` argument (`"SC"` for a symmetric
  circulant matrix or `"MST"` for a minimum spanning tree).

+ Replaced some inefficient parts of the C++ code.

+ Added a `pyproject.toml` for the package build.


## ccmmpy 0.1

+ Initial public release.
