"""Single-Spark resource checks. Never interrupts another workload."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import torch


def gpu_processes():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=10, check=True)
    rows = []
    for line in result.stdout.splitlines():
        pid, _, name = line.partition(',')
        if pid.strip().isdigit() and int(pid) != os.getpid():
            rows.append({'pid': int(pid), 'name': name.strip()})
    return rows


def available_host_bytes():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('Cannot determine unified-memory availability')


def require_idle_gpu(reserve_gib=16.):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; device work is pending')
    rows = gpu_processes()
    if rows:
        raise RuntimeError(f'GPU occupied; leave existing workloads running: {rows}')
    if available_host_bytes() < reserve_gib * 2**30:
        raise RuntimeError('Insufficient available host/unified memory')


def set_memory_limit(device, gib):
    if torch.device(device).type != 'cuda':
        return
    total = torch.cuda.get_device_properties(device).total_memory
    if total <= 0:
        raise RuntimeError('Cannot determine CUDA allocation limit')
    torch.cuda.set_per_process_memory_fraction(min(.7, gib * 2**30 / total), device)
