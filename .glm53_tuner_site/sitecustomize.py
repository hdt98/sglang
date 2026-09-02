"""Single-visible-GPU recovery shim for AITER's multiprocessing tuner.

The tuner builds a PID-to-GPU map when its process pool starts.  If an
isolated candidate faults and the pool replaces that worker, the replacement
PID is absent from the original map.  With exactly one visible GPU the mapping
is unambiguous, so add the replacement PID before entering the original worker
function.  This shim is only used by the isolated tuning command.
"""

import multiprocessing as mp

import torch

import aiter.utility.mp_tuner as _mp_tuner


_original_work_group = _mp_tuner.work_group


def _single_visible_gpu_work_group(gpu_id_map, *args, **kwargs):
    pid = mp.current_process().pid
    if pid not in gpu_id_map and torch.cuda.device_count() == 1:
        gpu_id_map = dict(gpu_id_map)
        gpu_id_map[pid] = 0
    return _original_work_group(gpu_id_map, *args, **kwargs)


_mp_tuner.work_group = _single_visible_gpu_work_group
