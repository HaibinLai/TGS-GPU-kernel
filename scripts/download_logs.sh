#!/usr/bin/env bash
set -euo pipefail

# Default configs
SRC_PATH="/tmp/cudalog"
DEST_BASE="./cuda_logs"
NAME_FILTER=""        # regex to filter container names, empty = all running containers
INTERVAL=""           # seconds; if set, loop forever with this interval
TIMESTAMP_FMT="+%Y%m%d_%H%M%S"

usage() {
  cat <<'USAGE'
Usage: fetch_cudalog.sh [-s SRC_PATH] [-d DEST_DIR] [-f NAME_REGEX] [-i SECONDS]

Options:
  -s  Source path in container to copy (default: /tmp/cudalog)
  -d  Destination base dir on host (default: ./cuda_logs)
  -f  Filter running containers by name (grep -E regex). Empty = all
  -i  Interval seconds. If set, loop pulling logs every N seconds

Examples:
  # 一次性从所有容器复制 /tmp/cudalog 到 ./cuda_logs/<name>-<ts>/
  ./fetch_cudalog.sh

  # 只从名字包含 job_ 的容器复制，目标改为 ./logs
  ./fetch_cudalog.sh -f 'job_' -d ./logs

  # 每 60 秒重复拉取（适合跑训练时持续同步）
  ./fetch_cudalog.sh -i 60
USAGE
}

while getopts ":s:d:f:i:h" opt; do
  case "$opt" in
    s) SRC_PATH="$OPTARG" ;;
    d) DEST_BASE="$OPTARG" ;;
    f) NAME_FILTER="$OPTARG" ;;
    i) INTERVAL="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 2 ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; usage; exit 2 ;;
  esac
done

copy_once() {
  local containers
  # 列出正在运行的容器：ID 和 Name
  mapfile -t containers < <(docker ps --format '{{.ID}} {{.Names}}')

  if [[ ${#containers[@]} -eq 0 ]]; then
    echo "[INFO] No running containers."
    return 0
  fi

  mkdir -p "$DEST_BASE"

  for line in "${containers[@]}"; do
    cid="${line%% *}"
    cname="${line#* }"

    # 名称过滤
    if [[ -n "$NAME_FILTER" ]] && ! grep -Eq "$NAME_FILTER" <<<"$cname"; then
      continue
    fi

    # 检查容器内路径是否存在
    if ! docker exec "$cid" bash -c "test -e \"$SRC_PATH\""; then
      echo "[WARN] $cname ($cid): $SRC_PATH not found, skip."
      continue
    fi

    ts="$(date "$TIMESTAMP_FMT")"
    dest_dir="$DEST_BASE/${cname}-${ts}"
    mkdir -p "$dest_dir"

    echo "[INFO] Copying from $cname ($cid): $SRC_PATH -> $dest_dir"

    # 用 tar 流方式复制，既适合文件也适合目录，并保留权限/时间戳
    parent="$(dirname "$SRC_PATH")"
    base="$(basename "$SRC_PATH")"

    if ! docker exec "$cid" bash -c "set -e; cd \"$parent\" && tar -cf - \"$base\"" \
        | tar -C "$dest_dir" -xf - ; then
      echo "[ERROR] Failed to copy from $cname ($cid)."
      continue
    fi

    echo "[OK]   Saved to: $dest_dir"
  done
}

if [[ -z "$INTERVAL" ]]; then
  copy_once
else
  # 循环模式
  while true; do
    echo "========== $(date '+%F %T') =========="
    copy_once
    sleep "$INTERVAL"
  done
fi
