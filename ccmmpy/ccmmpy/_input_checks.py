import numpy as np


def _check_array(array, ndim, input_name):
    if type(array) != np.ndarray:
        raise TypeError(
            "Expected numpy.ndarray, got {}.{} instead: {}={}"
            .format(type(array).__module__, type(array).__name__,
                    input_name, array)
        )

    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(
            "Expected dtype float, got {} instead: {}={}"
            .format(array.dtype, input_name, array)
        )

    if ndim == 2 and array.ndim != 2:
        raise ValueError(
            "Expected 2D array, got {}D array instead: {}={}"
            .format(array.ndim, input_name, array)
        )
    elif ndim == 1 and array.ndim != 1:
        raise ValueError(
            "Expected 1D array, got {}D array instead: {}={}"
            .format(array.ndim, input_name, array)
        )

    if np.isnan(array).any():
        raise ValueError(
            "Input contains NaN: {}={}".format(input_name, array)
        )

    return None


def _check_scalar(scalar, positive, input_name, upper_bound=None):
    if not type(scalar) in [int, float, np.float16, np.float32, np.float64,
                            np.int16, np.int32, np.int64]:
        raise TypeError(
            "Expected numerical value, got {} instead: {}={}"
            .format(type(scalar).__name__, input_name, scalar)
        )

    if positive and scalar <= 0:
        raise ValueError(
            "Expected positive scalar, got nonpositive instead: {}={}"
            .format(input_name, scalar)
        )
    elif scalar < 0:
        raise ValueError(
            "Expected nonnegative scalar, got negative instead: {}={}"
            .format(input_name, scalar)
        )

    if upper_bound is not None:
        if scalar >= upper_bound:
            raise ValueError(
                "Expected scalar below {}, instead: {}={}"
                .format(upper_bound, input_name, scalar)
            )

    return None


def _check_int(integer, positive, input_name):
    if not type(integer) in [int, np.int16, np.int32, np.int64]:
        raise TypeError(
            "Expected int, got {} instead: {}={}"
            .format(type(integer).__name__, input_name, integer)
        )

    if positive and integer <= 0:
        raise ValueError(
            "Expected positive integer, got nonpositive instead: {}={}"
            .format(input_name, integer)
        )
    elif integer < 0:
        raise ValueError(
            "Expected nonnegative integer, got negative instead: {}={}"
            .format(input_name, integer)
        )

    return None


def _check_boolean(boolean, input_name):
    if type(boolean) != bool:
        raise TypeError(
            "Expected bool, got {} instead: {}={}"
            .format(type(boolean).__name__, input_name, boolean)
        )

    return None


def _check_string(string, input_name):
    if type(string) != str:
        raise TypeError(
            "Expected string, got {} instead: {}={}"
            .format(type(string).__name__, input_name, string)
        )

    return None


def _check_weights(weights, input_name):
    from .sparseweights import SparseWeights

    if type(weights) != SparseWeights:
        raise TypeError(
            "Expected ccmmpy.weights.SparseWeights, got {}.{} instead: "
            "type({})={}"
            .format(type(weights).__module__, type(weights).__name__,
                    input_name, type(weights).__module__ + "." +
                    type(weights).__name__)
        )

    return None


def _check_lambdas(lambdas):
    _check_array(lambdas, 1, "lambdas")

    if (np.diff(lambdas) <= 0).sum() > 0:
        raise ValueError(
            "Expected monotonically increasing sequence for lambdas"
        )

    if (lambdas < 0).sum() > 0:
        raise ValueError(
            "Expected nonnegative values for lambdas"
        )

    return None


def _check_cluster_targets(low, high, n):
    _check_int(low, True, "target_low")
    _check_int(high, True, "target_high")

    if high > n:
        raise ValueError(
            "Expected target_high <= X.shape[0]"
        )

    if low > high:
        raise ValueError(
            "Expected target_low <= target_high"
        )

    return None


def _check_iterable(array, input_name):
    if not hasattr(array, "__iter__"):
        raise TypeError(
            "Expected an iterable object, got {} instead: {}={}"
            .format(array, input_name, array)
        )

    return None


def _check_string_value(string, input_name, options):
    _check_string(string, input_name)

    if string not in options:
        raise ValueError(
            "Expected one of ['SC', 'MST'] for {}, got {} instead"
            .format(input_name, string)
        )

    return None
