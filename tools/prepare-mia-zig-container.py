#!/usr/bin/env python3
"""Create an unstarted Mia native candidate from a reviewed host-local JSON spec.

No service is stopped or started. Credentials are mounted from a private file;
the retained original inspect and candidate request are private rollback records.
"""
import argparse
import importlib.util
import json
from pathlib import Path
from urllib.parse import quote


# The shared position pool both ranks hold, and the head's window: 786432 (768K), as the Python pair served before the
# 1M change. Memory does not back it: the compressed plane commits about 7 to 9 KB a position as a conversation fills
# it, on top of the 103 GiB model, and the guard's floor is reached near 200K positions on a GB10 pair. Past that the
# memory guard stops both ranks; the engine does not yet refuse a request for memory (see docs/mia-native-install).
# Default steady-state configuration aligned with MiaAI-Lab upstream + gpt-6.1-sol xhigh review:
# - 1048576 (1M) per-request window and shared KV pool, with 32768 (32K) default reply budget.
# - 3 concurrent streams (PARALLEL=3), 0.125 GiB (128 MiB) prompt cache snapshot budget, and xhigh reasoning effort.
DEFAULT_CONTEXT = 1048576
DEFAULT_POOL = 1048576
CONTEXT_STEP = 2048  # prompt chunk rows
MIN_CONTEXT, MAX_CONTEXT = 65536, 1048576
MAX_POOL = 2101248
DEFAULT_MAX_TOKENS = 32768
DEFAULT_PARALLEL = 3
DEFAULT_PROMPT_CACHE_GIB = 0.125
DEFAULT_REASONING_EFFORT = "xhigh"
DEFAULT_THINKING_BUDGET = 20000
DEFAULT_RDMA_IDLE_S = 2
DEFAULT_RDMA_NAP_US = 500
DEFAULT_SPILL_FLOOR_GIB = 50
DEFAULT_GRAPH_CAP = 128
DEFAULT_NCCL_MAX_NCHANNELS = 8
DEFAULT_ROUND_ROWS = 32
DEFAULT_COPY_ROWS = 32
DEFAULT_COPY_SELF_CAP = 15
DEFAULT_COPY_MULTI = 15
DEFAULT_PROMPT_PROGRESS_TOKENS = 65536
DEFAULT_ENGINE_ID = "tensorfold@629c524a0ea5+53d12a37f02a"
DEFAULT_RECIPE_PATCHES = "53d12a37f02a"
DEFAULT_MODEL_REVISION = (
    "Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1.1-2.9-bpw"
    "@aee640e2afba93be683bb5b9b420115fe7fbac20,quant=exl3-2.9bpw,"
    "ablit=1@d6681ac778ecbd7b422382ff6af88fb3d2debcb7"
)
DEFAULT_ENGRAM_REVISION = "deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170"


def parallel_streams(spec):
    """Concurrent engine streams (TF_DS_STREAMS, rank 0 --parallel, rank 1 --streams)."""
    parallel = spec.get("parallel", DEFAULT_PARALLEL)
    if type(parallel) is not int or not 1 <= parallel <= 16:
        raise ValueError("parallel must be an integer from 1 to 16")
    return parallel


def reasoning_effort(spec):
    """Default reasoning effort when a request names none (low, medium, high, xhigh)."""
    effort = spec.get("reasoning_effort", DEFAULT_REASONING_EFFORT)
    if not isinstance(effort, str) or effort not in ("low", "medium", "high", "xhigh"):
        raise ValueError("reasoning_effort must be low, medium, high or xhigh")
    return effort


def thinking_budget(spec):
    """Default reasoning token cap before the engine closes the think block (0 disables; PR #8 uses 20000)."""
    budget = spec.get("thinking_budget", DEFAULT_THINKING_BUDGET)
    if type(budget) is not int or not 0 <= budget <= MAX_CONTEXT // 2:
        raise ValueError("thinking_budget must be an integer from 0 to half the max context")
    return budget


def graphs_settings(spec):
    """(graphs, graph_cap): CUDA graphs and bounded graph-cache cap (0033) for both ranks."""
    graphs = spec.get("graphs", 1)
    if type(graphs) is not int or graphs not in (0, 1):
        raise ValueError("graphs must be 0 or 1")
    cap = spec.get("graph_cap", DEFAULT_GRAPH_CAP)
    if type(cap) is not int or not 0 <= cap <= 4096:
        raise ValueError("graph_cap must be an integer from 0 to 4096")
    return graphs, cap


def nccl_channels(spec):
    """NCCL_MAX_NCHANNELS cap (8 saves ~1.0 GiB pinned host memory per Spark over default 16; 0 leaves unset)."""
    ch = spec.get("nccl_max_nchannels", DEFAULT_NCCL_MAX_NCHANNELS)
    if type(ch) is not int or not 0 <= ch <= 64:
        raise ValueError("nccl_max_nchannels must be an integer from 0 to 64")
    return ch


def copy_window_settings(spec):
    """Wide copy-window settings (0035): (round_rows, copy_rows, copy_skip, copy_batch_skip, copy_self_cap, copy_multi)."""
    round_rows = spec.get("round_rows", DEFAULT_ROUND_ROWS)
    if type(round_rows) is not int or round_rows not in (16, 32, 48):
        raise ValueError("round_rows must be 16, 32 or 48")
    copy_rows = spec.get("copy_rows", min(DEFAULT_COPY_ROWS, round_rows))
    if type(copy_rows) is not int or not 16 <= copy_rows <= round_rows:
        raise ValueError("copy_rows must be an integer from 16 to round_rows")
    copy_skip = spec.get("copy_skip", True)
    if copy_skip not in (False, True, 0, 1):
        raise ValueError("copy_skip must be a boolean or 0/1")
    copy_batch_skip = spec.get("copy_batch_skip", True)
    if copy_batch_skip not in (False, True, 0, 1):
        raise ValueError("copy_batch_skip must be a boolean or 0/1")
    copy_self_cap = spec.get("copy_self_cap", DEFAULT_COPY_SELF_CAP)
    if type(copy_self_cap) is not int or not 0 <= copy_self_cap <= 31:
        raise ValueError("copy_self_cap must be an integer from 0 to 31")
    copy_multi = spec.get("copy_multi", DEFAULT_COPY_MULTI)
    if type(copy_multi) is not int or not 0 <= copy_multi <= 31:
        raise ValueError("copy_multi must be an integer from 0 to 31")
    return round_rows, copy_rows, bool(copy_skip), bool(copy_batch_skip), copy_self_cap, copy_multi


def prompt_progress_settings(spec):
    """Rolling prompt progress token interval (0036: 0 disables, else >= 8192 tokens)."""
    tokens = spec.get("prompt_progress_tokens", DEFAULT_PROMPT_PROGRESS_TOKENS)
    if type(tokens) is not int or (tokens != 0 and not 8192 <= tokens <= MAX_CONTEXT):
        raise ValueError("prompt_progress_tokens must be 0 or an integer from 8192 to max context")
    return tokens


def fill_settings(spec):
    """Piecewise prefill/cut-in is explicit so historical candidates retain their scheduling."""
    values = tuple(spec.get(key, 0) for key in ("fill_chunks", "fill_first", "fill_cut_in"))
    for key, value, limit in zip(("fill_chunks", "fill_first", "fill_cut_in"), values, (64, 64, MAX_CONTEXT)):
        if type(value) is not int or not 0 <= value <= limit:
            raise ValueError(f"{key} must be an integer from 0 to {limit}")
    if values[0] == 0 and any(values[1:]):
        raise ValueError("fill_first and fill_cut_in require fill_chunks")
    return values


def prompt_cache_budget(spec):
    """Kept prompt states' GPU snapshot budget in GiB (0.125 GiB = 128 MiB ~= 20 states, triggering LRU eviction)."""
    gib = spec.get("prompt_cache_gib", DEFAULT_PROMPT_CACHE_GIB)
    if type(gib) not in (int, float) or not 0.03125 <= gib <= 1.0 or gib != gib:
        raise ValueError("prompt_cache_gib must be from 0.03125 to 1.0 GiB")
    return gib


def window(spec):
    """(context, max_tokens): the head's window and default reply budget."""
    context = spec.get("context", DEFAULT_CONTEXT)
    if type(context) is not int or not MIN_CONTEXT <= context <= MAX_CONTEXT or context % CONTEXT_STEP:
        raise ValueError(f"context must be a multiple of {CONTEXT_STEP} from {MIN_CONTEXT} to {MAX_CONTEXT}")
    max_tokens = spec.get("max_tokens", min(DEFAULT_MAX_TOKENS, context // 2))
    if type(max_tokens) is not int or not 1024 <= max_tokens <= context // 2:
        raise ValueError("max_tokens must be from 1024 to half the context")
    return context, max_tokens


def shard_pool(context, round_rows=DEFAULT_ROUND_ROWS):
    """The pool KV_SHARD=1 needs so each of the two owner parts holds a whole CONTEXT window (2,101,248 at 1M)."""
    return ((context + round_rows + 2 + 2047) // 2048) * 2048 * 2


def kv_shard_enabled(spec):
    """Whether cross-rank KV sharding (0015/0018/0022) is enabled."""
    val = spec.get("kv_shard", False)
    if val not in (False, True, 0, 1):
        raise ValueError("kv_shard must be a boolean or 0/1")
    return bool(val)


def spill_settings(spec):
    """Return (spill_gib, spill_floor_gib, spill_dir) when Spill v2 (0013/0016/0019) is enabled, else (0, 0, None)."""
    gib = spec.get("spill_gib", 0)
    if type(gib) not in (int, float) or gib < 0 or gib > 1024 or gib != gib:
        raise ValueError("spill_gib must be a number from 0 to 1024 GiB")
    if gib == 0:
        return 0, 0, None
    floor = spec.get("spill_floor_gib", DEFAULT_SPILL_FLOOR_GIB)
    if type(floor) not in (int, float) or floor < 1 or floor > 1024 or floor != floor:
        raise ValueError("spill_floor_gib must be a number from 1 to 1024 GiB")
    spill_dir = spec.get("spill_dir")
    if not isinstance(spill_dir, str) or not spill_dir:
        raise ValueError("spill_dir is required when spill_gib > 0")
    path = Path(spill_dir)
    if not path.is_absolute() or ".." in path.parts or ":" in str(path):
        raise ValueError("spill_dir must be an absolute, unambiguous path")
    return gib, floor, path


def pool_positions(spec, context):
    """The shared pool both ranks hold (0008): defaults to shard_pool when kv_shard is on, else context."""
    sharded = kv_shard_enabled(spec)
    round_rows, *_ = copy_window_settings(spec)
    default_pool = shard_pool(context, round_rows) if sharded else (DEFAULT_POOL if "context" not in spec else context)
    pool = spec.get("pool", default_pool)
    if type(pool) is not int or not context <= pool <= MAX_POOL or pool % CONTEXT_STEP:
        raise ValueError(f"pool must be a multiple of {CONTEXT_STEP} from {context} to {MAX_POOL}")
    if sharded and (pool % 4096 != 0 or pool // 2 - (round_rows + 2) < context):
        raise ValueError(f"kv_shard pool must be a multiple of 4096 and at least {shard_pool(context, round_rows)}")
    return pool


def rdma_backoff(spec):
    """(idle_s, nap_us): RDMA gather proxy idle back-off settings (0010)."""
    idle_s = spec.get("rdma_idle_s", DEFAULT_RDMA_IDLE_S)
    if type(idle_s) not in (int, float) or not 0 <= idle_s <= 3600 or idle_s != idle_s:
        raise ValueError("rdma_idle_s must be from 0 to 3600 seconds")
    nap_us = spec.get("rdma_nap_us", DEFAULT_RDMA_NAP_US)
    if type(nap_us) is not int or not 1 <= nap_us <= 1000000:
        raise ValueError("rdma_nap_us must be an integer from 1 to 1000000 microseconds")
    return idle_s, nap_us


def engine_identity(spec, image_labels=None):
    """(engine_id, recipe_patches): prefers explicit spec overrides, else inspected image labels, else module defaults."""
    labels = image_labels or {}
    default_patches = labels.get("tf.patches") or DEFAULT_RECIPE_PATCHES
    tf_ref = (labels.get("tf.ref") or "629c524a0ea5")[:12]
    default_engine = f"tensorfold@{tf_ref}+{default_patches}" if labels.get("tf.patches") else DEFAULT_ENGINE_ID
    recipe_patches = spec.get("recipe_patches", default_patches)
    engine_id = spec.get("engine_id", default_engine)
    if not isinstance(recipe_patches, str) or not recipe_patches or not isinstance(engine_id, str) or not engine_id:
        raise ValueError("engine_id and recipe_patches must be non-empty strings")
    return engine_id, recipe_patches


def candidate_config(spec, image_id, image_labels=None):
    rank = spec["rank"]
    if type(rank) is not int or rank not in (0, 1):
        raise ValueError("rank must be 0 or 1")
    if not image_id.startswith("sha256:"):
        raise ValueError("Resolve the image to an immutable ID first")
    context, max_tokens = window(spec)
    sharded = kv_shard_enabled(spec)
    spill_gib, spill_floor, spill_dir = spill_settings(spec)
    round_rows, copy_rows, copy_skip, copy_batch_skip, copy_self_cap, copy_multi = copy_window_settings(spec)
    pool = pool_positions(spec, context)
    graphs, graph_cap = graphs_settings(spec)
    nchannels = nccl_channels(spec)
    progress_tokens = prompt_progress_settings(spec)
    fill_chunks, fill_first, fill_cut_in = fill_settings(spec)
    engine_id, recipe_patches = engine_identity(spec, image_labels)
    parallel = parallel_streams(spec)
    effort = reasoning_effort(spec)
    budget = thinking_budget(spec)
    cache_gib = prompt_cache_budget(spec)
    idle_s, nap_us = rdma_backoff(spec)
    gid = spec["gid_index"]
    if type(gid) is not int or not 0 <= gid <= 255:
        raise ValueError("Invalid verified RoCE GID index")
    for key in ("master_address", "socket_interface", "rdma_device"):
        value = spec[key]
        if not isinstance(value, str) or not value or any(c.isspace() for c in value):
            raise ValueError("Missing or invalid transport setting: " + key)
    port = spec.get("master_port", 29620)
    api_port = spec.get("api_port", 8012)
    if any(type(p) is not int or not 1024 <= p <= 65535 for p in (port, api_port)):
        raise ValueError("Invalid service port")
    rdma_nodes = spec["rdma_nodes"]
    if not isinstance(rdma_nodes, list) or not rdma_nodes:
        raise ValueError("Enumerate this host's InfiniBand character devices")
    if any(not isinstance(p, str) or Path(p).parent != Path("/dev/infiniband")
           or Path(p).name in (".", "..") for p in rdma_nodes):
        raise ValueError("Invalid InfiniBand device node")
    paths = {"model_dir": "/model", "cache_dir": "/rank-cache",
             "kit_dir": "/kit", "engram_dir": "/engram",
             "token_map": "/token-map.json"}
    if rank == 0:
        paths["api_key_file"] = "/run/secrets/tensorfold-api-key"
    binds = []
    for key, destination in paths.items():
        path = Path(spec[key])
        if not path.is_absolute() or ".." in path.parts or ":" in str(path):
            raise ValueError("Mount sources must be absolute, unambiguous paths")
        binds.append(f"{path}:{destination}:ro")
    if len({spec[k] for k in paths}) != len(paths):
        raise ValueError("Runtime artifacts must have separate paths")
    if spill_dir is not None:
        if str(spill_dir) in {str(Path(spec[k])) for k in paths}:
            raise ValueError("spill_dir must not overlap runtime artifact paths")
        binds.append(f"{spill_dir}:/spill:rw")
    env = {
        "TZ": "Asia/Taipei", "HF_HUB_OFFLINE": "1",
        "ENV": "", "BASH_ENV": "", "_CUDA_COMPAT_PATH": "/dev/null",
        "TF_DS_REPLAY_FLOOR": "1", "TF_DS_STREAMS": str(parallel),
        "TF_DS_ROUND_ROWS": str(round_rows),
        "TF_RDMA_IDLE_S": str(idle_s), "TF_RDMA_NAP_US": str(nap_us),
        "TF_RDMA_GID_INDEX": str(gid), "NCCL_IB_GID_INDEX": str(gid),
        "NCCL_IB_HCA": spec["rdma_device"], "NCCL_NET": "IB",
        "NCCL_IB_DISABLE": "0", "NCCL_SOCKET_IFNAME": spec["socket_interface"],
        "NCCL_DEBUG": "WARN",
    }
    if nchannels > 0:
        env["NCCL_MAX_NCHANNELS"] = str(nchannels)
    if fill_chunks:
        env.update(TF_FILL_CHUNKS=str(fill_chunks), TF_FILL_FIRST=str(fill_first),
                   TF_FILL_CUT_IN=str(fill_cut_in))
    if pool != context or "pool" in spec or sharded:
        env.update(TF_DS_POOL=str(pool), TF_DS_WINDOW=str(context))
    if sharded or spill_gib > 0:
        env.update(
            TF_DS_ENGINE_ID=engine_id,
            TF_DS_RECIPE_PATCHES=recipe_patches,
            TF_DS_MODEL_REVISION=spec.get("model_revision", DEFAULT_MODEL_REVISION),
            TF_DS_ENGRAM_REVISION=spec.get("engram_revision", DEFAULT_ENGRAM_REVISION),
        )
    if sharded:
        env.update(TF_DS_KV_SHARD="1", TF_DS_KV_SHARD_CARRY="1",
                   TF_DS_KV_SHARD_TWINS="1", TF_DS_KV_SHARD_CHECK="1024")
    if spill_gib > 0:
        env.update(TF_DS_SPILL_DIR="/spill", TF_DS_SPILL_GIB=str(spill_gib),
                   TF_DS_SPILL_MIN_FREE_GIB=str(spill_floor), TF_DS_SPILL_MIN_TOKENS="8192",
                   TF_DS_SPILL_QUOTA="0", TF_DS_SPILL_STAGING_MIB="128",
                   TF_SPILL_TURNS="4", TF_SPILL_FLUSH_MS="60000")
    if rank == 0:
        env.update(TF_TP_WORLD="2", TF_TP_MASTER=spec["master_address"],
                   TF_TP_PORT=str(port), TF_DS_RANK_CACHE="/rank-cache",
                   TF_DS_KIT="/kit", TENSORFOLD_CUDA_KERNELS="/kit/aot",
                   TF_DS_ENGRAM="/engram", TF_DS_TOKEN_MAP="/token-map.json",
                   TF_DS_GRAPHS=str(graphs), TF_DS_HC_SIDE="1", TF_DS_L2_PREFETCH="1",
                   TF_DS_ENGRAM_AIO="1", TF_RDMA_DEVICES=spec["rdma_device"],
                   TF_DS_PREFILL_INTERLEAVE_MS="250",
                   TF_DS_COPY_ROWS=str(copy_rows),
                   TF_DS_COPY_SKIP="1" if copy_skip else "0",
                   TF_DS_COPY_BATCH_SKIP="1" if copy_batch_skip else "0",
                   TF_DS_COPY_SELF_CAP=str(copy_self_cap),
                   TF_DS_COPY_MULTI=str(copy_multi))
        if graphs == 1 and graph_cap > 0:
            env["TF_DS_GRAPH_CAP"] = str(graph_cap)
        if progress_tokens > 0:
            env["TF_DS_PROMPT_PROGRESS_TOKENS"] = str(progress_tokens)
        command = ["tensorfold-native", "serve", "/model", "--backend", "cuda",
                   "--host", "0.0.0.0", "--port", str(api_port), "--name", "deepseek-chat",
                   "--context", str(context), "--parallel", str(parallel), "--max-tokens", str(max_tokens),
                   "--prompt-cache-gib", str(cache_gib),
                   "--temperature", "1.0", "--top-p", "0.95", "--top-k", "20",
                   "--thinking", "--reasoning-effort", effort]
        if budget > 0:
            command += ["--thinking-budget", str(budget)]
        command += ["--api-key-file", "/run/secrets/tensorfold-api-key"]
        if spill_gib > 0:
            command += ["--spill-dir", "/spill", "--spill-gib", str(spill_gib),
                        "--spill-min-free-gib", str(spill_floor), "--spill-min-tokens", "8192"]
    else:
        command = ["tf-dsv41-lanes", "/model", "/rank-cache", "1", "2",
                   spec["master_address"], str(port), "/kit", "--engram", "/engram",
                   "--token-map", "/token-map.json", "--pool", str(pool), "--streams", str(parallel),
                   "--round-rows", str(round_rows),
                   "--drafts", "1", "--graphs", str(graphs)]
        if graphs == 1 and graph_cap > 0:
            command += ["--graph-cap", str(graph_cap)]
        command += ["--side", "1", "--prefetch", "1", "--aio", "1", "--rdma", spec["rdma_device"]]
    return {
        "Image": image_id, "Entrypoint": [], "Cmd": command,
        "Env": [f"{k}={v}" for k, v in env.items()],
        "Labels": {"deployment.family": "mia-eaq-zig-ablit", "deployment.rank": str(rank)},
        "HostConfig": {
            "NetworkMode": "host", "IpcMode": "host", "AutoRemove": False,
            "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
            "Binds": binds, "CapAdd": ["IPC_LOCK"],
            "DeviceRequests": [{"Driver": "nvidia", "Count": -1,
                                "Capabilities": [["gpu"]]}],
            "Devices": [{"PathOnHost": p, "PathInContainer": p,
                         "CgroupPermissions": "rwm"} for p in rdma_nodes],
            "Ulimits": [{"Name": "memlock", "Soft": -1, "Hard": -1},
                        {"Name": "stack", "Soft": 67108864, "Hard": 67108864}],
            "LogConfig": {"Type": "json-file", "Config": {"max-size": "20m", "max-file": "3"}},
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    module_spec = importlib.util.spec_from_file_location(
        "prefix_container", Path(__file__).with_name("prepare-prefix-container.py"))
    docker = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(docker)
    spec = json.loads(args.spec.read_text())
    image = docker.inspect("image", args.image)
    image_labels = image["Config"].get("Labels", {})
    if image_labels.get("tf.engine") != "zig":
        raise ValueError("Expected the reviewed native Zig image")
    context, _ = window(spec)
    pool = pool_positions(spec, context)
    request = candidate_config(spec, image["Id"], image_labels)
    if any(not Path(p).is_char_device() for p in spec["rdma_nodes"]):
        raise ValueError("The enumerated RDMA devices are not available on this host")
    # Refuse missing local files before Docker can silently create empty directories.
    for bind in request["HostConfig"]["Binds"]:
        source, destination, _ = bind.split(":")
        path = Path(source)
        is_file = destination in ("/token-map.json", "/run/secrets/tensorfold-api-key")
        if not (path.is_file() if is_file else path.is_dir()):
            raise ValueError("Missing runtime artifact: " + destination)
        if destination == "/run/secrets/tensorfold-api-key":
            if path.stat().st_mode & 0o077 or not path.read_text().strip():
                raise ValueError("API key file must be nonempty and private")
    engram = Path(spec["engram_dir"])
    for name, size in (("model-00047-of-00048.safetensors", 101535150936),
                       ("model-00048-of-00048.safetensors", 101537926640)):
        if (engram / name).stat().st_size != size:
            raise ValueError("Engram files do not match the 1017 release")
    original = docker.inspect("container", args.original)
    args.state_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    docker.save(args.state_dir / "original.json", original)
    docker.save(args.state_dir / "request.json", request)
    connection = docker.Docker()
    connection.request("POST", "/containers/create?name=" + quote(args.name, safe=""),
                       json.dumps(request), {"Content-Type": "application/json"})
    response = connection.getresponse()
    result = json.loads(response.read())
    connection.close()
    if response.status != 201:
        raise RuntimeError("Candidate creation failed; inspect private Docker records")
    created = docker.inspect("container", result["Id"])
    if created["State"]["Running"] or created["Config"]["Cmd"] != request["Cmd"]:
        raise RuntimeError("Unexpected candidate state; original retained")
    plan = {"original_name": args.original, "original_id": original["Id"],
            "candidate_name": args.name, "candidate_id": result["Id"],
            "image": image["Id"], "rank": spec["rank"], "context": context, "pool": pool}
    docker.save(args.state_dir / "plan.json", plan)
    print(json.dumps(plan))


if __name__ == "__main__":
    main()
