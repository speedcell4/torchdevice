import os
import socket
import subprocess
from pathlib import Path
from typing import List

from filelock import FileLock


def occupy(devices: List[int]):
    import torch

    for index in devices:
        _ = torch.empty((1,), device=torch.device(f'cuda:{index}'))


def set_cuda_visible_devices(n: int = 1) -> None:
    CUDA_VISIBLE_DEVICES = 'CUDA_VISIBLE_DEVICES'
    cuda_visible_devices = os.environ.get(CUDA_VISIBLE_DEVICES, '').strip()

    if CUDA_VISIBLE_DEVICES in os.environ:
        try:
            return occupy([int(device) for device in cuda_visible_devices.split(',')])
        except ValueError:
            print(f'ignore existing {CUDA_VISIBLE_DEVICES} = {cuda_visible_devices}')

    with FileLock(str(Path.home() / f'.{CUDA_VISIBLE_DEVICES}.{socket.gethostname()}')):
        statuses = subprocess.check_output('nvidia-smi -q -d PIDS | grep Processes', shell=True)
        devices = [device for device, status in enumerate(statuses.splitlines()) if b': None' in status]

        if len(devices) < n:
            raise RuntimeError(f'the number of free devices ({devices})) are not enough < {n}')

        os.environ[CUDA_VISIBLE_DEVICES] = ','.join(map(str, devices[:n]))
        print(f'{CUDA_VISIBLE_DEVICES} <- {os.environ[CUDA_VISIBLE_DEVICES]}')
        return occupy(devices)
