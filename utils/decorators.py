import functools
import time

benchmark_stack = {}

def measure_time(func):

    @functools.wraps(func)
    def time_measurement(*args, **kwargs):

        t_start = time.perf_counter()
        ret = func(*args, **kwargs)
        exe_time = time.perf_counter() - t_start

        global benchmark_stack
        if not(func.__qualname__ in benchmark_stack):
            benchmark_stack[func.__qualname__] = []
        benchmark_stack[func.__qualname__].append(exe_time)
        print(f"Process time of {func.__qualname__}: {exe_time} s")

        return ret

    return time_measurement