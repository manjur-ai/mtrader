import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(func):
        return func


@njit
def monotonic_stack_for_value1_lessthan_value2(values1, values2):
    n = len(values1)
    result = np.full(n, -1, dtype=np.int32)
    stack = np.empty(n, dtype=np.int32)
    stack_ptr = 0

    for i in range(n - 1, -1, -1):
        while stack_ptr > 0 and values1[stack[stack_ptr - 1]] >= values2[i]:
            stack_ptr -= 1
        if stack_ptr > 0:
            result[i] = stack[stack_ptr - 1]
        stack[stack_ptr] = i
        stack_ptr += 1

    return result


@njit
def monotonic_stack_for_value1_gt_value2(values1, values2):
    n = len(values1)
    result = np.full(n, -1, dtype=np.int32)
    stack = np.empty(n, dtype=np.int32)
    stack_ptr = 0

    for i in range(n - 1, -1, -1):
        while stack_ptr > 0 and values1[stack[stack_ptr - 1]] <= values2[i]:
            stack_ptr -= 1
        if stack_ptr > 0:
            result[i] = stack[stack_ptr - 1]
        stack[stack_ptr] = i
        stack_ptr += 1

    return result
