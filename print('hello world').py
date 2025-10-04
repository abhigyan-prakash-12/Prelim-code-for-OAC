print('hello world')

import numpy as np

def memory_limit_test(max_size=1_000_000_000, step=10_000_000):
    """
    Gradually allocate larger arrays until Python is killed or memory error occurs.
    
    Parameters:
        max_size (int): Maximum number of elements to try.
        step (int): Number of elements to increase per iteration.
    """
    size = step
    try:
        while size <= max_size:
            print(f"Testing array with {size} elements...")
            arr = np.ones(size, dtype=np.float64)  # allocate array
            arr *= 2  # simple computation to test memory usage
            size += step
    except MemoryError:
        print(f"MemoryError reached at {size} elements")
        return size
    
memory_limit_test() 