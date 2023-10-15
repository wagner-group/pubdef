import contextlib
import random

import numpy as np


@contextlib.contextmanager
def np_temp_seed(seed):
    np_state = np.random.get_state()
    rand_state = random.getstate()
    np.random.seed(seed)
    random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(rand_state)
