import copy
import importlib.util
from pathlib import Path
import unittest


module_spec = importlib.util.spec_from_file_location(
    "mia_zig", Path(__file__).parents[1] / "tools/prepare-mia-zig-container.py")
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)


class NativeCandidateTests(unittest.TestCase):
    def setUp(self):
        self.spec = dict(rank=0, master_address="192.0.2.1", socket_interface="rdma0",
                         rdma_device="roce0", gid_index=3, model_dir="/srv/mia/model",
                         cache_dir="/srv/mia/ranks", kit_dir="/srv/mia/kit",
                         engram_dir="/srv/engram", token_map="/srv/mia/token-map.json",
                         api_key_file="/srv/private/api-key",
                         rdma_nodes=["/dev/infiniband/rdma_cm", "/dev/infiniband/uverbs0"])

    def test_head_preserves_endpoint_and_highest_thinking(self):
        original = copy.deepcopy(self.spec)
        cfg = module.candidate_config(self.spec, "sha256:test")
        cmd = cfg["Cmd"]
        self.assertEqual(cmd[:3], ["tensorfold-native", "serve", "/model"])
        for flag, value in (("--context", "1048576"), ("--parallel", "3"),
                            ("--max-tokens", "32768"), ("--prompt-cache-gib", "0.125"),
                            ("--name", "deepseek-chat"),
                            ("--reasoning-effort", "xhigh"), ("--thinking-budget", "20000"),
                            ("--port", "8012")):
            self.assertEqual(cmd[cmd.index(flag) + 1], value)
        self.assertIn("--thinking", cmd)
        self.assertIn("--api-key-file", cmd)
        self.assertNotIn("--no-drafts", cmd)
        self.assertEqual(self.spec, original)
        no_budget = module.candidate_config({**self.spec, "thinking_budget": 0}, "sha256:test")["Cmd"]
        self.assertNotIn("--thinking-budget", no_budget)

    def test_worker_uses_same_pool_streams_and_drafts(self):
        self.spec["rank"] = 1
        cfg = module.candidate_config(self.spec, "sha256:test")
        cmd = cfg["Cmd"]
        self.assertEqual(cmd[:5], ["tf-dsv41-lanes", "/model", "/rank-cache", "1", "2"])
        for flag, value in (("--pool", "1048576"), ("--streams", "3"),
                            ("--drafts", "1"), ("--rdma", "roce0")):
            self.assertEqual(cmd[cmd.index(flag) + 1], value)
        self.assertFalse(any("api-key" in bind for bind in cfg["HostConfig"]["Binds"]))

    def test_head_window_and_worker_pool_share_one_context_when_overridden(self):
        spec = {**self.spec, "context": 262144, "max_tokens": 65536, "parallel": 4, "reasoning_effort": "high"}
        head = module.candidate_config(spec, "sha256:test")["Cmd"]
        worker = module.candidate_config({**spec, "rank": 1}, "sha256:test")["Cmd"]
        self.assertEqual(head[head.index("--context") + 1], "262144")
        self.assertEqual(head[head.index("--max-tokens") + 1], "65536")
        self.assertEqual(head[head.index("--parallel") + 1], "4")
        self.assertEqual(head[head.index("--reasoning-effort") + 1], "high")
        self.assertEqual(worker[worker.index("--pool") + 1], "262144")
        self.assertEqual(worker[worker.index("--streams") + 1], "4")
        self.assertEqual(module.window({}), (1048576, 32768))
        self.assertEqual(module.pool_positions({}, 1048576), 1048576)

    def test_pool_larger_than_window_stays_consistent_across_both_ranks(self):
        spec = {**self.spec, "context": 524288, "pool": 1048576, "max_tokens": 131072}
        head_cfg = module.candidate_config(spec, "sha256:test")
        worker_cfg = module.candidate_config({**spec, "rank": 1}, "sha256:test")
        head_env = dict(item.split("=", 1) for item in head_cfg["Env"])
        worker_env = dict(item.split("=", 1) for item in worker_cfg["Env"])
        self.assertEqual(head_cfg["Cmd"][head_cfg["Cmd"].index("--context") + 1], "524288")
        self.assertEqual(worker_cfg["Cmd"][worker_cfg["Cmd"].index("--pool") + 1], "1048576")
        for env in (head_env, worker_env):
            self.assertEqual(env["TF_DS_POOL"], "1048576")
            self.assertEqual(env["TF_DS_WINDOW"], "524288")

    def test_kv_shard_and_spill_align_across_both_ranks(self):
        spec = {**self.spec, "kv_shard": True, "spill_gib": 32, "spill_dir": "/srv/mia/spill"}
        head_cfg = module.candidate_config(spec, "sha256:test")
        worker_cfg = module.candidate_config({**spec, "rank": 1}, "sha256:test")
        head_env = dict(item.split("=", 1) for item in head_cfg["Env"])
        worker_env = dict(item.split("=", 1) for item in worker_cfg["Env"])
        self.assertEqual(worker_cfg["Cmd"][worker_cfg["Cmd"].index("--pool") + 1], "2101248")
        self.assertEqual(head_cfg["Cmd"][head_cfg["Cmd"].index("--spill-gib") + 1], "32")
        self.assertEqual(head_cfg["Cmd"][head_cfg["Cmd"].index("--spill-dir") + 1], "/spill")
        for cfg, env in ((head_cfg, head_env), (worker_cfg, worker_env)):
            self.assertIn("/srv/mia/spill:/spill:rw", cfg["HostConfig"]["Binds"])
            self.assertEqual(env["TF_DS_POOL"], "2101248")
            self.assertEqual(env["TF_DS_WINDOW"], "1048576")
            self.assertEqual(env["TF_DS_KV_SHARD"], "1")
            self.assertEqual(env["TF_DS_SPILL_GIB"], "32")
            self.assertEqual(env["TF_DS_SPILL_DIR"], "/spill")
            self.assertTrue(env["TF_DS_ENGINE_ID"].startswith("tensorfold@"))
            self.assertIn("EAQ-v1.1-2.9-bpw@aee640e2", env["TF_DS_MODEL_REVISION"])
            self.assertEqual(env["ENV"], "")
            self.assertEqual(env["BASH_ENV"], "")
            self.assertEqual(env["_CUDA_COMPAT_PATH"], "/dev/null")

    def test_both_ranks_require_rdma_and_read_only_artifacts(self):
        for rank in (0, 1):
            self.spec["rank"] = rank
            cfg = module.candidate_config(self.spec, "sha256:test")
            env = dict(item.split("=", 1) for item in cfg["Env"])
            self.assertEqual(env["TF_RDMA_GID_INDEX"], "3")
            self.assertEqual(env["TF_RDMA_IDLE_S"], "2")
            self.assertEqual(env["TF_RDMA_NAP_US"], "500")
            self.assertEqual(env["NCCL_IB_GID_INDEX"], "3")
            self.assertEqual(env["NCCL_NET"], "IB")
            self.assertEqual(env["NCCL_IB_HCA"], "roce0")
            self.assertEqual(env["NCCL_MAX_NCHANNELS"], "8")
            self.assertEqual(env["TF_DS_ROUND_ROWS"], "32")
            self.assertNotIn("TENSORFOLD_MEMORY_RESERVE_GIB", env)
            self.assertNotIn("WEIGHTLESS_STEER_PATH", env)
            host = cfg["HostConfig"]
            self.assertEqual(host["NetworkMode"], "host")
            self.assertEqual(host["RestartPolicy"]["Name"], "no")
            self.assertFalse(host["AutoRemove"])
            self.assertTrue(all(b.endswith(":ro") for b in host["Binds"]))
            self.assertIn("/dev/infiniband", host["Devices"][0]["PathOnHost"])

    def test_graph_cap_wide_copy_and_progress_defaults_and_image_labels(self):
        labels = {"tf.patches": "deadbeef1234", "tf.ref": "629c524a0ea57512a5342abb498d1b7c25353343"}
        spec = {**self.spec, "kv_shard": True, "context": 786432}
        head_cfg = module.candidate_config(spec, "sha256:test", labels)
        worker_cfg = module.candidate_config({**spec, "rank": 1}, "sha256:test", labels)
        head_env = dict(item.split("=", 1) for item in head_cfg["Env"])
        worker_env = dict(item.split("=", 1) for item in worker_cfg["Env"])
        self.assertEqual(head_env["TF_DS_GRAPHS"], "1")
        self.assertEqual(head_env["TF_DS_GRAPH_CAP"], "128")
        self.assertEqual(head_env["TF_DS_ROUND_ROWS"], "32")
        self.assertEqual(head_env["TF_DS_COPY_ROWS"], "32")
        self.assertEqual(head_env["TF_DS_COPY_SKIP"], "1")
        self.assertEqual(head_env["TF_DS_COPY_BATCH_SKIP"], "1")
        self.assertEqual(head_env["TF_DS_COPY_SELF_CAP"], "15")
        self.assertEqual(head_env["TF_DS_COPY_MULTI"], "15")
        self.assertEqual(head_env["TF_DS_PROMPT_PROGRESS_TOKENS"], "65536")
        self.assertEqual(head_env["TF_DS_POOL"], "1576960")
        self.assertEqual(head_env["TF_DS_RECIPE_PATCHES"], "deadbeef1234")
        self.assertEqual(head_env["TF_DS_ENGINE_ID"], "tensorfold@629c524a0ea5+deadbeef1234")
        self.assertEqual(worker_env["TF_DS_RECIPE_PATCHES"], "deadbeef1234")
        self.assertEqual(worker_env["TF_DS_ROUND_ROWS"], "32")
        wcmd = worker_cfg["Cmd"]
        self.assertEqual(wcmd[wcmd.index("--round-rows") + 1], "32")
        self.assertEqual(wcmd[wcmd.index("--graphs") + 1], "1")
        self.assertEqual(wcmd[wcmd.index("--graph-cap") + 1], "128")
        # graphs=0 omits graph_cap on both ranks; prompt_progress_tokens=0 omits TF_DS_PROMPT_PROGRESS_TOKENS
        off_head = module.candidate_config({**self.spec, "graphs": 0, "prompt_progress_tokens": 0}, "sha256:test")
        off_worker = module.candidate_config({**self.spec, "rank": 1, "graphs": 0}, "sha256:test")
        off_env = dict(item.split("=", 1) for item in off_head["Env"])
        self.assertEqual(off_env["TF_DS_GRAPHS"], "0")
        self.assertNotIn("TF_DS_GRAPH_CAP", off_env)
        self.assertNotIn("TF_DS_PROMPT_PROGRESS_TOKENS", off_env)
        self.assertEqual(off_worker["Cmd"][off_worker["Cmd"].index("--graphs") + 1], "0")
        self.assertNotIn("--graph-cap", off_worker["Cmd"])

    def test_piecewise_prefill_is_explicit_and_shared_by_both_ranks(self):
        for rank in (0, 1):
            legacy = module.candidate_config({**self.spec, "rank": rank}, "sha256:test")
            self.assertFalse(any(e.startswith("TF_FILL_") for e in legacy["Env"]))
            cfg = module.candidate_config({**self.spec, "rank": rank, "fill_chunks": 1,
                                           "fill_first": 1, "fill_cut_in": 8192}, "sha256:test")
            env = dict(e.split("=", 1) for e in cfg["Env"])
            self.assertEqual({k: env[k] for k in ("TF_FILL_CHUNKS", "TF_FILL_FIRST", "TF_FILL_CUT_IN")},
                             {"TF_FILL_CHUNKS": "1", "TF_FILL_FIRST": "1", "TF_FILL_CUT_IN": "8192"})

    def test_rejects_unsafe_or_mismatched_inputs(self):
        for key, value in (("rank", 2), ("rank", True), ("gid_index", -1),
                           ("gid_index", 256), ("context", 1000), ("context", 131073),
                           ("context", 2097152), ("context", True), ("context", "131072"),
                           ("pool", 131072), ("pool", 786433), ("pool", 4194304),
                           ("pool", True), ("pool", "1048576"),
                           ("graphs", 2), ("graphs", True), ("graphs", "1"),
                           ("graph_cap", -1), ("graph_cap", 5000), ("graph_cap", True),
                           ("nccl_max_nchannels", -1), ("nccl_max_nchannels", 65), ("nccl_max_nchannels", True),
                           ("round_rows", 15), ("round_rows", 24), ("round_rows", 33), ("round_rows", True),
                           ("copy_rows", 15), ("copy_rows", 33), ("copy_rows", True),
                           ("copy_self_cap", -1), ("copy_self_cap", 32), ("copy_multi", 32),
                           ("prompt_progress_tokens", 4096), ("prompt_progress_tokens", -1), ("prompt_progress_tokens", True),
                           ("fill_chunks", True), ("fill_chunks", -1), ("fill_chunks", 65),
                           ("fill_first", 1), ("fill_cut_in", 8192),
                           ("rdma_idle_s", -0.1), ("rdma_idle_s", 3600.1),
                           ("rdma_idle_s", True), ("rdma_idle_s", "2"),
                           ("rdma_nap_us", 0), ("rdma_nap_us", 1000001),
                           ("rdma_nap_us", True), ("rdma_nap_us", 1.5),
                           ("prompt_cache_gib", 0), ("prompt_cache_gib", -0.1),
                           ("prompt_cache_gib", 2.0), ("prompt_cache_gib", True),
                           ("prompt_cache_gib", "0.125"),
                           ("parallel", 0), ("parallel", 17), ("parallel", True), ("parallel", "3"),
                           ("reasoning_effort", "max"), ("reasoning_effort", ""), ("reasoning_effort", 1),
                           ("thinking_budget", -1), ("thinking_budget", True), ("thinking_budget", "20000"),
                           ("thinking_budget", 600000),
                           ("max_tokens", 600000), ("max_tokens", 512), ("master_port", 80),
                           ("api_port", 70000), ("socket_interface", ""),
                           ("rdma_device", "roce 0"), ("model_dir", "relative"),
                           ("model_dir", "/srv/../model"), ("cache_dir", "/srv/mia/model"),
                           ("token_map", "/srv/token:map"), ("rdma_nodes", []),
                           ("rdma_nodes", ["/dev/null"])):
            spec = {**self.spec, key: value}
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                module.candidate_config(spec, "sha256:test")
        with self.assertRaises(ValueError):
            module.candidate_config(self.spec, "mutable:latest")


if __name__ == "__main__":
    unittest.main()
