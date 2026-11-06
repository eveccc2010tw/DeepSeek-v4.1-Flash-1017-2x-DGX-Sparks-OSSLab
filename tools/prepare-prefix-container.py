#!/usr/bin/env python3
"""Prepare an unstarted prefix-cache container, preserving the current deployment.

Run on each Docker host after reviewing/pushing the source patch. This creates
private rollback records and a candidate; it never stops or starts a service.
"""
import argparse
import copy
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
from urllib.parse import quote


class Docker(http.client.HTTPConnection):
    def __init__(self):
        super().__init__("localhost", timeout=60)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect("/var/run/docker.sock")


def inspect(kind, name):
    return json.loads(subprocess.check_output(
        ["docker", kind, "inspect", name], text=True))[0]


def save(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def candidate_config(old, image_id, *, prefill_chunk=None, model_dir=None,
                     cache_dir=None, ib_gid_index=None, vision=False, steering_alpha=None,
                     no_steering=False, context_tokens=None, vision_extra=None):
    """Return a candidate without mutating the retained rollback specification."""
    cfg, host = copy.deepcopy(old["Config"]), copy.deepcopy(old["HostConfig"])
    cmd = cfg["Cmd"]
    if (cmd[cmd.index("--parallel") + 1] != "1"
            or cmd[cmd.index("--context") + 1] not in ("786432", "1048576")
            or cmd[cmd.index("--tp") + 1] != "2"):
        raise ValueError("Expected a 768K or 1M, parallel=1, TP2 deployment")
    if context_tokens is not None:
        if context_tokens not in (786432, 1048576):
            raise ValueError("Expected context 786432 or 1048576 tokens")
        cmd[cmd.index("--context") + 1] = str(context_tokens)
    if host["NetworkMode"] != "host" or host.get("AutoRemove"):
        raise ValueError("Expected host networking and retained containers")
    updates = {"TF_DS_SINGLE_PREFIX": "1", "TF_DS_KEEP": "1",
               "TF_DS_KEEP_ENTRIES": "1", "TF_DS_KEEP_SHRINK": "0", "TF_DS_SPILL_GIB": "0"}
    if vision_extra is not None:
        extra = Path(vision_extra)
        if not extra.is_absolute() or ".." in extra.parts:
            raise ValueError("Vision extra must be an absolute container directory")
        updates["TF_DS_VISION_EXTRA"] = str(extra)
    if no_steering:
        if steering_alpha is not None:
            raise ValueError("no_steering and steering_alpha are mutually exclusive")
        updates.update(WEIGHTLESS_STEER_PATH="", WEIGHTLESS_STEER_ALPHA="0")
    if steering_alpha is not None:
        if steering_alpha not in (0, 0.75):
            raise ValueError("Expected steering alpha 0 or 0.75")
        updates["WEIGHTLESS_STEER_ALPHA"] = "0" if steering_alpha == 0 else "0.75"
    if prefill_chunk is not None:
        if prefill_chunk not in (512, 1024):
            raise ValueError("Expected prefill chunk 512 or 1024")
        updates["TF_DS_PREFILL_CHUNK"] = str(prefill_chunk)
    if ib_gid_index is not None:
        if not 0 <= ib_gid_index <= 255:
            raise ValueError("Invalid IB GID index")
        updates["NCCL_IB_GID_INDEX"] = str(ib_gid_index)
    if (model_dir is None) != (cache_dir is None):
        raise ValueError("A replacement model requires a separate new cache directory")
    if model_dir is not None:
        replacements = {"/model": str(model_dir), "/cache": str(cache_dir)}
        if any(not Path(p).is_absolute() for p in replacements.values()):
            raise ValueError("Model and cache paths must be absolute")
        if replacements["/model"] == replacements["/cache"]:
            raise ValueError("Model and cache directories must differ")
        old_mounts = {m["Destination"]: m["Source"] for m in old["Mounts"]}
        if replacements["/cache"] == old_mounts.get("/cache"):
            raise ValueError("A replacement model must not reuse the old cache directory")
        binds, seen = [], []
        for bind in host.get("Binds") or []:
            parts = bind.split(":")
            if len(parts) not in (2, 3):
                raise ValueError("Unsupported bind mount format")
            destination = parts[1]
            if destination in replacements:
                seen.append(destination)
                mode = "ro" if destination == "/model" else "rw"
                bind = f"{replacements[destination]}:{destination}:{mode}"
            binds.append(bind)
        if sorted(seen) != ["/cache", "/model"]:
            raise ValueError("Expected exactly one model and one cache bind mount")
        host["Binds"] = binds
    if vision and "--vision" not in cmd:
        cmd.append("--vision")
    env = dict(item.split("=", 1) for item in cfg["Env"])
    env.update(updates)
    cfg.update(Image=image_id, Env=[f"{k}={v}" for k, v in env.items()])
    return cfg, host, updates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--container", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--state-dir", type=Path, required=True)
    p.add_argument("--prefill-chunk", type=int, choices=(512, 1024),
                   help="Override the prefill chunk on both ranks; otherwise preserve the current setting")
    p.add_argument("--model-dir", type=Path, help="Replacement model directory; requires a fresh --cache-dir")
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--ib-gid-index", type=int, help="Verified host-specific RoCE v2 GID index")
    p.add_argument("--vision", action="store_true", help="Enable image input on both ranks")
    p.add_argument("--vision-extra", help="Container directory containing separate VL routing biases")
    p.add_argument("--context-tokens", type=int, choices=(786432, 1048576),
                   help="Override the context limit while preserving weights, caches and other settings")
    steering = p.add_mutually_exclusive_group()
    steering.add_argument("--steering-alpha", type=float, choices=(0, 0.75),
                   help="Use 0 for a projection-disabled control; preserve all other settings")
    steering.add_argument("--no-steering", action="store_true",
                          help="Disable the GLP loader for an independently prepared weight overlay")
    args = p.parse_args()
    old = inspect("container", args.container)
    image = inspect("image", args.image)
    cfg, host, updates = candidate_config(
        old, image["Id"], prefill_chunk=args.prefill_chunk, model_dir=args.model_dir,
        cache_dir=args.cache_dir, ib_gid_index=args.ib_gid_index, vision=args.vision,
        steering_alpha=args.steering_alpha, no_steering=args.no_steering,
        context_tokens=args.context_tokens, vision_extra=args.vision_extra)
    if args.model_dir is not None:
        if not args.model_dir.is_dir():
            raise ValueError("Replacement model directory does not exist")
        args.cache_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    state = args.state_dir.resolve()
    state.mkdir(mode=0o700, parents=True, exist_ok=False)
    save(state / "original.json", old)
    suffix = "-model-candidate" if args.model_dir is not None else "-prefix-candidate"
    if args.model_dir is None and args.steering_alpha is not None:
        suffix = "-steering-off-candidate" if args.steering_alpha == 0 else "-steering-glp-candidate"
    if args.context_tokens is not None:
        suffix += f"-context-{args.context_tokens}"
    candidate = args.container + suffix
    save(state / "request.json", {**cfg, "HostConfig": host})
    connection = Docker()
    connection.request("POST", "/containers/create?name=" + quote(candidate, safe=""),
                       json.dumps({**cfg, "HostConfig": host}),
                       {"Content-Type": "application/json"})
    response = connection.getresponse()
    result = json.loads(response.read())
    connection.close()
    if response.status != 201:
        raise RuntimeError(f"Docker create failed ({response.status}); inspect private records")
    plan = {"original_id": old["Id"], "original_name": args.container,
            "original_image": old["Image"], "candidate_id": result["Id"],
            "candidate_name": candidate, "candidate_image": image["Id"],
            "cache_env": updates,
            "context_tokens": int(cfg["Cmd"][cfg["Cmd"].index("--context") + 1])}
    save(state / "plan.json", plan)
    created = inspect("container", result["Id"])
    if created["State"]["Running"] or created["Config"]["Cmd"] != cfg["Cmd"]:
        raise RuntimeError("Unexpected candidate state; original is still untouched")
    print(json.dumps(plan))


if __name__ == "__main__":
    main()
