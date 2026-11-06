import importlib.util
import json
from pathlib import Path
import subprocess
import unittest

spec = importlib.util.spec_from_file_location(
    "native_pair", Path(__file__).parents[1] / "tools/supervise-native-pair.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class NativePairTests(unittest.TestCase):
    def setUp(self):
        self.nodes = [dict(host=f"host{r}", container=f"native{r}", operations_dir=f"/srv/rank{r}")
                      for r in (0, 1)]
        self.calls = []
        self.states = [False, False]
        self.available = [113, 113]

    def execute(self, host, args):
        self.calls.append((host, args))
        rank = int(host[-1])
        if args[:4] == ["docker", "inspect", "--format", "{{json .Config.Labels}}"]:
            return json.dumps({"deployment.family": "mia-eaq-zig-ablit", "deployment.rank": str(rank)})
        if args[:2] == ["docker", "inspect"]:
            return str(self.states[rank]).lower()
        if args == ["cat", "/proc/meminfo"]:
            return f"MemAvailable: {int(self.available[rank] * 1048576)} kB\n"
        if args[:2] == ["docker", "start"]:
            self.states[rank] = True
        elif args[:2] == ["docker", "stop"]:
            self.states[rank] = False
        return ""

    def pair(self, sleep=lambda _: None):
        return module.Pair(self.nodes, self.execute, sleep)

    def test_refuses_either_host_below_memory_gate_without_starting(self):
        for rank in (0, 1):
            self.calls = []; self.available = [113, 113]; self.available[rank] = 109.99
            with self.assertRaisesRegex(RuntimeError, "110 GiB"):
                self.pair().start()
            self.assertFalse(any(args[:2] == ["docker", "start"] for _, args in self.calls))

    def test_starts_both_guards_before_worker_then_head(self):
        self.pair().start()
        actions = [(host, args) for host, args in self.calls if args[0] != "cat" and args[1] != "inspect"]
        self.assertEqual([args[1] for _, args in actions], ["start"] * 4)
        self.assertTrue(actions[0][1][0].endswith("memguard.sh"))
        self.assertEqual(actions[0][1][2:], ["native0", "2.5", "/srv/rank0/memguard.log", "/srv/rank0/memguard.pid", "4", "1.0"])
        self.assertEqual([host for host, args in actions if args[0] == "docker"], ["host1", "host0"])

    def test_guard_supports_per_node_transition_overrides(self):
        self.nodes[0]["memguard_floor_gib"] = "1.0"
        self.nodes[0]["memguard_hard_gib"] = "0.6"
        self.pair().start()
        guards = [args for _, args in self.calls if args[0].endswith("memguard.sh") and args[1] == "start"]
        self.assertEqual(guards[0][2:], ["native0", "1.0", "/srv/rank0/memguard.log", "/srv/rank0/memguard.pid", "4", "0.6"])
        self.assertEqual(guards[1][2:], ["native1", "2.5", "/srv/rank1/memguard.log", "/srv/rank1/memguard.pid", "4", "1.0"])

    def test_existing_pair_is_adopted_without_requiring_free_weight_memory(self):
        self.states = [True, True]; self.available = [4, 4]
        self.pair().start()
        self.assertFalse(any(args[:2] == ["docker", "start"] or args[0] == "cat" for _, args in self.calls))
        self.assertEqual(sum(args[0].endswith("memguard.sh") for _, args in self.calls), 2)

    def test_peer_exit_stops_both_ranks_and_guards(self):
        self.states = [True, True]
        def sleep(_): self.states[1] = False
        with self.assertRaisesRegex(RuntimeError, "rank exited"):
            self.pair(sleep).run()
        self.assertEqual(self.states, [False, False])
        self.assertEqual(sum(args[:2] == ["docker", "stop"] for _, args in self.calls), 2)
        self.assertEqual(sum(args[0].endswith("memguard.sh") and args[1] == "stop" for _, args in self.calls), 2)

    def test_partial_pair_is_stopped_before_memory_check_and_restart(self):
        self.states = [True, False]
        self.pair().start()
        first_stop = next(i for i, (_, args) in enumerate(self.calls) if args[:2] == ["docker", "stop"])
        first_memory = next(i for i, (_, args) in enumerate(self.calls) if args[0] == "cat")
        self.assertLess(first_stop, first_memory)
        self.assertEqual(self.states, [True, True])

    def test_transport_failures_are_bounded(self):
        probes = 0
        def disconnected(host, args):
            nonlocal probes
            probes += 1
            raise subprocess.CalledProcessError(255, ["tmssh"])
        pair = self.pair(); pair.execute = disconnected
        with self.assertRaisesRegex(RuntimeError, "three checks"):
            pair.watch()
        self.assertEqual(probes, 3)

    def test_unrelated_container_is_rejected_without_stop_or_guard(self):
        def original(host, args):
            self.calls.append((host, args))
            return json.dumps({"deployment.family": "original"})
        pair = self.pair(); pair.execute = original
        with self.assertRaisesRegex(ValueError, "outside this native pair"):
            pair.run()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][1][:2], ["docker", "inspect"])

    def test_failed_rank_stop_keeps_its_memory_watchdog(self):
        self.states = [True, True]
        def failed_stop(host, args):
            if host == "host0" and args[:2] == ["docker", "stop"]:
                self.calls.append((host, args))
                raise subprocess.CalledProcessError(1, ["docker", "stop"])
            return self.execute(host, args)
        pair = self.pair(); pair.execute = failed_stop
        with self.assertRaises(RuntimeError):
            pair.stop()
        self.assertEqual(self.states, [True, False])
        guard_stops = [host for host, args in self.calls
                       if args[0].endswith("memguard.sh") and args[1] == "stop"]
        self.assertEqual(guard_stops, ["host1"])


if __name__ == "__main__":
    unittest.main()
