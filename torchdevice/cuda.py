import os
import socket
import subprocess
from pathlib import Path
from typing import List

from filelock import FileLock


def set_cuda_visible_devices(n: int = 1) -> List[int]:
    CUDA_VISIBLE_DEVICES = 'CUDA_VISIBLE_DEVICES'

    if CUDA_VISIBLE_DEVICES in os.environ:
        return [int(device) for device in os.environ[CUDA_VISIBLE_DEVICES].split(',')]

    with FileLock(str(Path.home() / f'.{CUDA_VISIBLE_DEVICES}.{socket.gethostname()}')):
        statuses = subprocess.check_output('nvidia-smi -q -d PIDS | grep Processes', shell=True)
        devices = [device for device, status in enumerate(statuses.splitlines()) if b': None' in status]

        if len(devices) < n:
            raise RuntimeError(f'the number of free devices ({devices})) are not enough < {n}')

        os.environ[CUDA_VISIBLE_DEVICES] = ','.join(map(str, devices[:n]))
        print(f'{CUDA_VISIBLE_DEVICES} <- {os.environ[CUDA_VISIBLE_DEVICES]}')

        import torch

        for index in range(n):
            _ = torch.empty((1,), device=torch.device(f'cuda:{index}'))

    return devices
