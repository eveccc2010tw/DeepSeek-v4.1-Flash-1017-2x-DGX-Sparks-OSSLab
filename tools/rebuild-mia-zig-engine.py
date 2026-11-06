#!/usr/bin/env python3
"""Rebuild only the native engine of a Mia native image with extra TensorFold patches.

The engine compiles inside a memory-capped container started from the existing image, which already holds the
recipe's Zig, CUDA toolchain and inputs tools. A production pair on the same host keeps its memory: the build's own
cgroup is OOM-killed before the host is. The compile steps match the recipe's zig/Dockerfile; only the parallelism
follows the container's CPU set. The new image is the base image plus the rebuilt /opt/tensorfold engine files.
Nothing is started, stopped or removed.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

TF_REPO = "https://github.com/bertholomus/TensorFold.git"
TF_REF = "fbbc97132cd004237d9887b8d3185896efaedc5c"
INPUTS_REF = "59402086381a42c74d53053443224eec2790b4c5"
# Files of /opt/tensorfold that come from the recipe's inputs checkout, not from the engine build.
INPUTS_FILES = {"bin/tf-dsv41-rank-cache", "share/doc/tensorfold/LICENSE-inputs-branch"}
INPUTS_PREFIX = "tools/inputs/"
OPTIONAL_ADDED_FILES = {"bin/tf-dsv41-port"}

ENGINE_SCRIPT = r"""set -euo pipefail
git init -q /tensorfold && cd /tensorfold && git remote add origin "$TF_REPO"
git fetch -q --depth 1 origin "$TF_REF" && git checkout -q --detach FETCH_HEAD
test "$(git rev-parse HEAD)" = "$TF_REF"
for p in /opt/tf-patches/*.patch; do echo "applying $p"; git apply --check "$p" && git apply "$p"; done
/opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm="$SM" -Doptimize=ReleaseFast \
  --prefix /opt/tensorfold -j"$(nproc)" install native
test -x /opt/tensorfold/native/bin/tensorfold-native && test -x /opt/tensorfold/bin/tf-dsv41-lanes
mkdir -p /opt/tensorfold/share/doc/tensorfold
for f in LICENSE NOTICE ATTRIBUTION.md THIRD_PARTY_NOTICES.md LICENSES; do
  if [ -e "$f" ]; then cp -a "$f" /opt/tensorfold/share/doc/tensorfold/; fi
done
if [ -e tools/dsv41-zig/ATTRIBUTION.md ]; then
  cp -a tools/dsv41-zig/ATTRIBUTION.md /opt/tensorfold/share/doc/tensorfold/DSV41-ATTRIBUTION.md
fi
if [ -d /opt/tf-inputs-patches ] && [ -n "${INPUTS_REF:-}" ]; then
  git init -q /tf-inputs && cd /tf-inputs && git remote add origin "$TF_REPO"
  git fetch -q --depth 1 origin "$INPUTS_REF" && git checkout -q --detach FETCH_HEAD
  test "$(git rev-parse HEAD)" = "$INPUTS_REF"
  for p in /opt/tf-inputs-patches/*.patch; do
    [ -e "$p" ] || continue
    echo "applying inputs patch $p"
    git apply --check "$p" && git apply "$p"
  done
  /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm="$SM" -Doptimize=ReleaseFast \
    --prefix /opt/inputs -j"$(nproc)" install
  test -x /opt/inputs/bin/tf-dsv41-rank-cache
  cp /opt/inputs/bin/tf-dsv41-rank-cache /opt/tensorfold/bin/
  mkdir -p /opt/tensorfold/tools/inputs
  cp tools/dsv41-zig/inputs/make_token_map.py tools/dsv41-zig/kit/make_rope.py tools/dsv41-zig/kit/make_bias_vl.py \
     /opt/tensorfold/tools/inputs/
  cp LICENSE /opt/tensorfold/share/doc/tensorfold/LICENSE-inputs-branch
fi
chown -R "$OUT_OWNER" /opt/tensorfold
"""


def patch_files(patch_dir):
    patches = sorted(Path(patch_dir).glob("*.patch"))
    if not patches:
        raise SystemExit(f"no *.patch in {patch_dir}")
    return patches


def patches_hash(patches, repo=TF_REPO, ref=TF_REF):
    """The patch set and engine source as one 12-character label, in the recipe's filename order."""
    digest = hashlib.sha256()
    for p in patches:
        digest.update(Path(p).read_bytes())
    digest.update(f"{repo} {ref}\n".encode())
    return digest.hexdigest()[:12]


def build_argv(base, patch_dir, out_dir, memory, cpus, owner, repo=TF_REPO, ref=TF_REF, sm="121",
               inputs_patch_dir=None, inputs_ref=INPUTS_REF):
    """docker run for the engine compile: swap equals the memory cap, lowest CPU weight, a fresh /opt/tensorfold."""
    argv = ["docker", "run", "--rm", "--name", "mia-engine-rebuild", "--network", "host",
            "--memory", memory, "--memory-swap", memory, "--cpuset-cpus", cpus, "--cpu-shares", "2",
            "-e", f"TF_REPO={repo}", "-e", f"TF_REF={ref}", "-e", f"SM={sm}", "-e", f"OUT_OWNER={owner}",
            "-v", f"{Path(patch_dir).resolve()}:/opt/tf-patches:ro",
            "-v", f"{Path(out_dir).resolve()}:/opt/tensorfold"]
    if inputs_patch_dir is not None:
        argv += ["-e", f"INPUTS_REF={inputs_ref}",
                 "-v", f"{Path(inputs_patch_dir).resolve()}:/opt/tf-inputs-patches:ro"]
    return argv + ["--entrypoint", "bash", base, "-c", ENGINE_SCRIPT]


def check_files(base_files, new_files):
    """The rebuilt engine must replace engine files only: nothing unexpected, and only inputs files left from the base."""
    base_files, new_files = set(base_files), set(new_files)
    added = sorted(new_files - base_files - OPTIONAL_ADDED_FILES)
    kept = sorted(f for f in base_files - new_files if f not in INPUTS_FILES and not f.startswith(INPUTS_PREFIX))
    if added or kept:
        raise SystemExit(f"engine file set changed: added={added} missing={kept}")
    return sorted(new_files)


def dockerfile(base, labels):
    """BuildKit resolves a local image by name, not by a bare image ID; the caller pins the ID around the build."""
    if base.startswith("sha256:"):
        raise ValueError("FROM needs the base image's name; BuildKit treats a bare ID as a registry repository")
    lines = [f"FROM {base}", "COPY tensorfold/ /opt/tensorfold/"]
    lines += [f"LABEL {k}={json.dumps(v)}" for k, v in sorted(labels.items())]
    return "\n".join(lines) + "\n"


def run(argv, **kw):
    return subprocess.run(argv, check=True, text=True, **kw)


def image_id(ref):
    return run(["docker", "image", "inspect", ref, "--format", "{{.Id}}"], capture_output=True).stdout.strip()


def compiled(marker, out, built):
    """A finished compile for the same base image and patch set can be reused; anything else is rebuilt."""
    if not marker.exists() or not out.is_dir() or not any(out.iterdir()):
        return False
    return json.loads(marker.read_text()) == built


def image_files(image):
    out = run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "find", image,
               "/opt/tensorfold", "-type", "f,l", "-printf", "%P\\n"], capture_output=True).stdout
    return [line for line in out.splitlines() if line]


def capabilities(image):
    out = run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "tensorfold-native", image,
               "capabilities", "--json"], capture_output=True).stdout
    return json.loads(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-image", required=True)
    ap.add_argument("--patch-dir", required=True, type=Path)
    ap.add_argument("--inputs-patch-dir", type=Path, default=None)
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--memory", default="4g")
    ap.add_argument("--cpus", default="0-3")
    ap.add_argument("--allow-capabilities-diff", action="store_true")
    args = ap.parse_args()

    patches = patch_files(args.patch_dir)
    inputs_patches = patch_files(args.inputs_patch_dir) if args.inputs_patch_dir else []
    all_patches = list(patches) + list(inputs_patches)
    base_id = image_id(args.base_image)
    out = args.work_dir / "context" / "tensorfold"
    marker = args.work_dir / "engine-built.json"
    built = {"base_image": base_id, "tf.patches": patches_hash(all_patches)}
    if compiled(marker, out, built):
        print(f"reusing the engine compiled for {built}", flush=True)
    else:
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{out} is not empty and was not compiled for this base and patch set")
        out.mkdir(parents=True, exist_ok=True)
        run(build_argv(base_id, args.patch_dir, out, args.memory, args.cpus, f"{os.getuid()}:{os.getgid()}",
                       inputs_patch_dir=args.inputs_patch_dir))
        marker.write_text(json.dumps(built) + "\n")

    new_files = [str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() or p.is_symlink()]
    files = check_files(image_files(base_id), new_files)
    labels = {"tf.patches": patches_hash(all_patches), "tf.ref": TF_REF, "tf.engine": "zig",
              "tf.inputs": INPUTS_REF, "osslab.engine-rebuild-from": base_id,
              "osslab.engine-patches": ",".join(p.name for p in patches),
              "osslab.engine-build-memory": args.memory, "osslab.engine-build-cpus": args.cpus}
    if inputs_patches:
        labels["osslab.inputs-patches"] = ",".join(p.name for p in inputs_patches)
    (args.work_dir / "context" / "Dockerfile").write_text(dockerfile(args.base_image, labels))
    if image_id(args.base_image) != base_id:
        raise SystemExit(f"{args.base_image} moved during the rebuild")
    run(["docker", "build", "--network", "none", "-t", args.tag, str(args.work_dir / "context")])
    new_id = run(["docker", "image", "inspect", args.tag, "--format", "{{.Id}}"], capture_output=True).stdout.strip()

    base_caps, new_caps = capabilities(base_id), capabilities(new_id)
    summary = {"base_image": base_id, "image": new_id, "tag": args.tag, "labels": labels,
               "patches": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in patches},
               "inputs_patches": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs_patches},
               "engine_files": len(files), "capabilities_equal": base_caps == new_caps}
    (args.work_dir / "rebuild-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if base_caps != new_caps and not args.allow_capabilities_diff:
        sys.exit("capabilities differ from the base image")


if __name__ == "__main__":
    main()
