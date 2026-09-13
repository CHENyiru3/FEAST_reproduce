#!/usr/bin/env bash
set -euo pipefail

# Download individual MOSTA mouse embryo slice h5ad files.
#
# Sources:
# - Dataset page: https://db.cngb.org/stomics/datasets/STDS0000058/data
# - Download page: https://db.cngb.org/stomics/mosta/download/
#
# This script intentionally downloads only per-slice *.MOSTA.h5ad files.
# It excludes Mouse_embryo_all_stage.h5ad and non-slice derived h5ad files.
#
# Usage:
#   bash scripts/download_mosta_single_slice_h5ad.sh [output_dir]
#
# Check links and remote sizes without downloading:
#   CHECK_ONLY=1 bash scripts/download_mosta_single_slice_h5ad.sh
#
# Optional:
#   MANIFEST=/path/to/raw_urls.tsv bash scripts/download_mosta_single_slice_h5ad.sh
#   JOBS=4 bash scripts/download_mosta_single_slice_h5ad.sh

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dataset_dir="${DATASET_DIR:-${script_dir}}"
out_dir="${1:-${dataset_dir}/samples}"
check_only="${CHECK_ONLY:-0}"
jobs="${JOBS:-1}"
manifest="${MANIFEST:-${dataset_dir}/manifest/raw_urls.tsv}"
log_dir="${dataset_dir}/logs"
log_file="${log_dir}/download_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$log_dir"
exec > >(tee -a "$log_file") 2>&1

command -v curl >/dev/null 2>&1 || {
    echo "ERROR: curl is required." >&2
    exit 1
}

if [[ ! -f "$manifest" ]]; then
    echo "ERROR: manifest not found: $manifest" >&2
    exit 1
fi

if ! [[ "$jobs" =~ ^[0-9]+$ ]] || (( jobs < 1 )); then
    echo "ERROR: JOBS must be a positive integer." >&2
    exit 1
fi

human_bytes() {
    local bytes="$1"
    awk -v b="$bytes" 'BEGIN {
        split("B KiB MiB GiB TiB", unit)
        i = 1
        while (b >= 1024 && i < 5) {
            b /= 1024
            i++
        }
        printf "%.2f %s", b, unit[i]
    }'
}

remote_size() {
    local url="$1"
    curl -fsSIL --retry 3 --retry-delay 5 "$url" |
        tr -d '\r' |
        awk 'tolower($1) == "content-length:" {print $2}' |
        tail -n 1
}

download_or_check() {
    local sample="$1"
    local data_type="$2"
    local key="$3"
    local expected_bytes="$4"
    local url="$5"
    local target="${out_dir}/${sample}/${key}"

    if [[ "$check_only" == "1" ]]; then
        local bytes
        bytes="$(remote_size "$url")"
        printf '[check] %-14s %-8s %12s bytes  %s\n' "$sample" "$data_type" "${bytes:-unknown}" "$key"
        if [[ "$bytes" =~ ^[0-9]+$ ]]; then
            total_remote_bytes=$((total_remote_bytes + bytes))
        fi
        if [[ "$expected_bytes" =~ ^[0-9]+$ && "$bytes" != "$expected_bytes" ]]; then
            echo "ERROR: expected ${expected_bytes} bytes but remote reports ${bytes:-missing} for ${key}" >&2
            exit 1
        fi
        return
    fi

    mkdir -p "$(dirname "$target")"
    if [[ -f "$target" && "$expected_bytes" =~ ^[0-9]+$ ]]; then
        local existing_bytes
        existing_bytes="$(wc -c < "$target" | tr -d ' ')"
        if [[ "$existing_bytes" == "$expected_bytes" ]]; then
            echo "[skip] already complete: ${target}"
            return
        fi
    fi

    printf '[download] %-14s %-8s %s\n' "$sample" "$data_type" "$target"
    curl -fsSL \
        --retry 5 \
        --retry-delay 10 \
        --retry-connrefused \
        --retry-all-errors \
        --connect-timeout 30 \
        --speed-limit 102400 \
        --speed-time 60 \
        -C - \
        -o "$target" \
        "$url"

    if [[ "$expected_bytes" =~ ^[0-9]+$ ]]; then
        local local_bytes
        local_bytes="$(wc -c < "$target" | tr -d ' ')"
        if [[ "$local_bytes" != "$expected_bytes" ]]; then
            echo "ERROR: downloaded ${local_bytes} bytes but expected ${expected_bytes} bytes for ${target}" >&2
            exit 1
        fi
    fi
}

total=0
total_remote_bytes=0

if [[ "$check_only" == "1" || "$jobs" == "1" ]]; then
    while IFS=$'\t' read -r sample data_type key expected_bytes url; do
        url="${url%$'\r'}"
        [[ -z "${sample:-}" || "$sample" == "SampleID" ]] && continue
        total=$((total + 1))
        download_or_check "$sample" "$data_type" "$key" "$expected_bytes" "$url"
    done < "$manifest"
else
    echo "Mode: parallel download with ${jobs} jobs"
    active=0
    failed=0
    while IFS=$'\t' read -r sample data_type key expected_bytes url; do
        url="${url%$'\r'}"
        [[ -z "${sample:-}" || "$sample" == "SampleID" ]] && continue
        total=$((total + 1))
        (
            download_or_check "$sample" "$data_type" "$key" "$expected_bytes" "$url"
        ) &
        active=$((active + 1))
        if (( active >= jobs )); then
            if ! wait -n; then
                failed=1
            fi
            active=$((active - 1))
        fi
    done < "$manifest"

    while (( active > 0 )); do
        if ! wait -n; then
            failed=1
        fi
        active=$((active - 1))
    done

    if (( failed != 0 )); then
        echo "ERROR: one or more downloads failed." >&2
        exit 1
    fi
fi

if [[ "$check_only" == "1" ]]; then
    echo "Finished: checked ${total} per-slice MOSTA h5ad files."
    echo "Remote total: ${total_remote_bytes} bytes ($(human_bytes "$total_remote_bytes"))."
else
    echo "Finished: processed ${total} per-slice MOSTA h5ad files."
fi
echo "Log: ${log_file}"
