#!/usr/bin/env python3
"""Supervise two prepared native ranks through the managed tmssh gateway.

The private JSON config contains two host/container/operations_dir records.
This tool never creates containers, changes weights, or accesses API credentials.
Use a service manager with bounded restart attempts. Each host runs upstream's
memory watchdog; a confirmed rank exit stops the peer as well.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import signal
import subprocess
import time


def remote(host, argv):
    return subprocess.check_output(["tmssh", host, shlex.join(argv)],
                                   text=True, timeout=90, stderr=subprocess.DEVNULL).strip()


class Pair:
    def __init__(self, nodes, execute=remote, sleep=time.sleep):
        if len(nodes) != 2:
            raise ValueError("Exactly two ranks are required")
        if nodes[0]["host"] == nodes[1]["host"]:
            raise ValueError("The ranks require separate managed hosts")
        for node in nodes:
            if not node["host"] or not node["container"]:
                raise ValueError("Missing managed host or container")
            path = Path(node["operations_dir"])
            if not path.is_absolute() or ".." in path.parts:
                raise ValueError("operations_dir must be an absolute host-local path")
        self.nodes, self.execute, self.sleep = nodes, execute, sleep

    def call(self, rank, *args):
        return self.execute(self.nodes[rank]["host"], list(args))

    def running(self, rank):
        return self.call(rank, "docker", "inspect", "--format", "{{.State.Running}}",
                         self.nodes[rank]["container"]) == "true"

    def validate_targets(self):
        for rank in (0, 1):
            labels = json.loads(self.call(rank, "docker", "inspect", "--format",
                                         "{{json .Config.Labels}}", self.nodes[rank]["container"]))
            if (labels.get("deployment.family") != "mia-eaq-zig-ablit"
                    or labels.get("deployment.rank") != str(rank)):
                raise ValueError("Refusing to manage a container outside this native pair")

    def guard(self, rank, action):
        node = self.nodes[rank]
        root = Path(node["operations_dir"])
        args = [str(root / "memguard.sh"), action]
        if action == "start":
            floor = str(node.get("memguard_floor_gib", "2.5"))
            hold = str(node.get("memguard_hold_s", "4"))
            hard = str(node.get("memguard_hard_gib", "1.0"))
            args += [node["container"], floor, str(root / "memguard.log"),
                     str(root / "memguard.pid"), hold, hard]
        else:
            args += [str(root / "memguard.pid")]
        return self.call(rank, *args)

    def stop(self):
        errors, stopped = [], []
        for rank in (0, 1):
            try:
                self.call(rank, "docker", "stop", "-t", "30", self.nodes[rank]["container"])
                stopped.append(rank)
            except (subprocess.SubprocessError, OSError) as error:
                errors.append(error)
        # A failed stop may leave the model running. Its watchdog must outlive
        # this supervisor until a later confirmed stop can safely remove it.
        for rank in stopped:
            try:
                self.guard(rank, "stop")
            except (subprocess.SubprocessError, OSError) as error:
                errors.append(error)
        if errors:
            raise RuntimeError("Could not stop every rank/guard; inspect private host status")

    def start(self):
        states = [self.running(rank) for rank in (0, 1)]
        if all(states):
            for rank in (0, 1):
                self.guard(rank, "start")
            return
        if any(states):
            self.stop()
        # MemAvailable is in KiB. Read on both hosts immediately before launch;
        # 110 GiB allows clean page-cache variance (~111.5 GiB on rank 1) while
        # still refusing launch if another large workload holds memory.
        for rank in (0, 1):
            memory = self.call(rank, "cat", "/proc/meminfo")
            available = next(int(line.split()[1]) for line in memory.splitlines()
                             if line.startswith("MemAvailable:"))
            if available < 110 * 1024 * 1024:
                raise RuntimeError(f"Rank {rank} has less than 110 GiB available")
        for rank in (0, 1):
            self.guard(rank, "start")
        self.call(1, "docker", "start", self.nodes[1]["container"])
        self.sleep(2)
        self.call(0, "docker", "start", self.nodes[0]["container"])

    def watch(self):
        failures = 0
        while True:
            self.sleep(30)
            try:
                states = [self.running(rank) for rank in (0, 1)]
            except (subprocess.SubprocessError, OSError):
                failures += 1
                if failures < 3:
                    continue
                raise RuntimeError("Managed host status unavailable for three checks") from None
            failures = 0
            if not all(states):
                raise RuntimeError("A native rank exited; stopping the pair")

    def run(self):
        # Validate before entering the cleanup scope: an incorrect name must
        # never stop an unrelated original container.
        self.validate_targets()
        try:
            self.start()
            self.watch()
        finally:
            self.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    # Prevent a manual invocation and the service manager from owning the pair together.
    with args.config.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def interrupted(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        try:
            Pair(config["nodes"]).run()
        except KeyboardInterrupt:
            return
        except Exception as error:
            detail = str(error) if isinstance(error, (ValueError, RuntimeError)) else "inspect private host status"
            raise SystemExit(type(error).__name__ + ": " + detail) from None


if __name__ == "__main__":
    main()
