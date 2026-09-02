"""Explicit single-visible-GPU worker replacement support for AITER tuning."""

import multiprocessing as mp

import torch

import aiter.utility.mp_tuner as _mp_tuner


_original_work_group = _mp_tuner.work_group


def work_group(gpu_id_map, *args, **kwargs):
    pid = mp.current_process().pid
    if pid not in gpu_id_map and torch.cuda.device_count() == 1:
        gpu_id_map = dict(gpu_id_map)
        gpu_id_map[pid] = 0
    return _original_work_group(gpu_id_map, *args, **kwargs)
