#!/usr/bin/env bash
# Run on the Spark itself. Does not SSH, alter drivers, or launch paid compute.
# Image pull is the only required network operation; training has no network access.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-accept}"; if (($#)); then shift; fi
case "$ACTION" in doctor|accept|accept-small|sample|prepare|train|profile|report|shell) ;; *) echo 'Usage: bash spark/run.sh {accept|sample|prepare|train|profile|report} [args...]' >&2; exit 2;; esac
if [[ "$ACTION" == doctor ]]; then exec python3 "$ROOT/energy_spark/doctor.py" "$@"; fi
if [[ "$(uname -m)" != aarch64 && "$(uname -m)" != arm64 ]]; then
  echo 'This launcher is for the Arm-based DGX Spark. CPU diagnostics: python -m energy_spark.accept --device cpu --cpu-diagnostic --out runs/cpu' >&2; exit 2
fi
command -v docker >/dev/null || { echo 'Docker must be installed and the NVIDIA container runtime configured.' >&2; exit 2; }
docker info >/dev/null 2>&1 || { echo 'Docker is unavailable to this user. Resolve Docker access using the DGX Spark setup guide; this script does not modify groups.' >&2; exit 2; }
mkdir -p "$ROOT/runs" "$ROOT/local_data"
# NVIDIA's current Spark fine-tuning playbook names 25.11. This is an explicit
# known playbook starting point, NOT the newest available container. Keep its
# bundled PyTorch/CUDA intact. A locally validated newer NGC image can be supplied.
TAG="${ENERGY_IMAGE:-nvcr.io/nvidia/pytorch:25.11-py3}"
LOCK="$ROOT/runs/image.lock"
ACCEPTANCE="${ENERGY_ACCEPTANCE:-/runs/acceptance/acceptance.json}"
if [[ -s "$LOCK" && -z "${ENERGY_IMAGE:-}" ]]; then
  IMAGE="$(cat "$LOCK")"
  docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull --platform linux/arm64 "$IMAGE"
else
  docker pull --platform linux/arm64 "$TAG"
  IMAGE="$(docker image inspect "$TAG" --format '{{index .RepoDigests 0}}')"
  [[ "$IMAGE" == *@sha256:* ]] || { echo 'Could not resolve an immutable image digest.' >&2; exit 2; }
  printf '%s\n' "$IMAGE" > "$LOCK"
fi
[[ "$(docker image inspect "$IMAGE" --format '{{.Architecture}}')" == arm64 ]] || { echo 'Container architecture is not ARM64.' >&2; exit 2; }
printf 'Using pinned image: %s\n' "$IMAGE"
# Use only this workspace, no home-directory tokens, SSH keys, Docker socket or host IPC.
BASE=(docker run --rm --init --gpus device=0 --user "$(id -u):$(id -g)" --network none
 --cap-drop ALL --security-opt no-new-privileges --read-only --shm-size=2g
 --tmpfs /tmp:rw,exec,nosuid,size=4g -e HOME=/tmp -e PYTHONPATH=/workspace -e PYTHONDONTWRITEBYTECODE=1
 -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1
 -e "ENERGY_IMAGE_DIGEST=$IMAGE" -v "$ROOT:/workspace:ro" -v "$ROOT/runs:/runs:rw" -v "$ROOT/local_data:/data:rw"
 -w /workspace "$IMAGE")
case "$ACTION" in
 accept)       exec "${BASE[@]}" python -m energy_spark.accept --device cuda --include-large --out /runs/acceptance "$@" ;;
 accept-small) exec "${BASE[@]}" python -m energy_spark.accept --device cuda --out /runs/acceptance-small "$@" ;;
 sample)       exec "${BASE[@]}" python -m energy_spark.sample --device cuda "$@" ;;
 prepare)      exec "${BASE[@]}" python -m energy_spark.data "$@" ;;
 train)        exec "${BASE[@]}" python -m energy_spark.train --device cuda --acceptance "$ACCEPTANCE" "$@" ;;
 profile)      exec "${BASE[@]}" python -m energy_spark.diagnose profile --device cuda --acceptance "$ACCEPTANCE" "$@" ;;
 report)       exec "${BASE[@]}" python -m energy_spark.diagnose summarize --runs /runs --out /runs/SUMMARY.json "$@" ;;
 shell)        echo 'No interactive privileged shell is provided; use documented diagnostic subcommands.' >&2; exit 2 ;;
esac
