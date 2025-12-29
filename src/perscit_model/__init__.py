"""Perseus Citation Model package."""

import multiprocessing

# Set multiprocessing start method to 'spawn' to avoid fork() issues with CUDA
# and multi-threaded libraries like PyTorch. This prevents deadlocks and
# "Cannot re-initialize CUDA in forked subprocess" errors.
# NOTE: Don't use force=True - it causes reentrant ResourceTracker cleanup issues
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    # Start method already set (e.g., by another import), ignore
    pass

# Also set for multiprocess library (fork of multiprocessing used by some libraries)
try:
    import multiprocess
    multiprocess.set_start_method('spawn')
except (ImportError, RuntimeError):
    # multiprocess not installed or start method already set
    pass
