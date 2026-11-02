"""Conservative N607 capacity checks for the preregistered source experiments."""

import os
import subprocess
import time


def select_gpu(inventory, processes, *, minimum_free_mb=12288, max_visible_processes=2):
    if (not inventory or len({row[0] for row in inventory}) != len(inventory)
            or len({row[1] for row in inventory}) != len(inventory)
            or any(index < 0 or not uuid or free < 0 for index, uuid, free in inventory)):
        raise ValueError('A valid unique GPU inventory is required')
    counts = {uuid: set() for _, uuid, _ in inventory}
    for uuid, pid in processes:
        if uuid not in counts or pid <= 0:
            raise ValueError('Every visible compute process must belong to an inventoried GPU')
        counts[uuid].add(pid)
    for index, uuid, free in sorted(inventory):
        if free >= minimum_free_mb and len(counts[uuid]) < max_visible_processes:
            return index
    return None


def available_gpu():
    def query(fields):
        command = ['nvidia-smi', fields, '--format=csv,noheader,nounits']
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.splitlines()
    raw = query('--query-gpu=index,uuid,memory.free')
    jobs = query('--query-compute-apps=gpu_uuid,pid')
    inventory = []
    for row in raw:
        index, uuid, free = (part.strip() for part in row.split(','))
        inventory.append((int(index), uuid, int(free)))
    processes = []
    for row in jobs:
        if row.strip():
            uuid, pid = (part.strip() for part in row.split(','))
            processes.append((uuid, int(pid)))
    return select_gpu(inventory, processes)


def wait_for_gpu(status):
    while True:
        load = os.getloadavg()[0]
        gpu = available_gpu()
        if load <= os.cpu_count()-24 and gpu is not None:
            return gpu
        status('WAITING_COMPUTE_CAPACITY', load=load, available_gpu=gpu)
        time.sleep(60)


def wait_for_cpu(status):
    while os.getloadavg()[0] > os.cpu_count()-24:
        status('WAITING_CPU_CAPACITY', load=os.getloadavg()[0])
        time.sleep(60)
