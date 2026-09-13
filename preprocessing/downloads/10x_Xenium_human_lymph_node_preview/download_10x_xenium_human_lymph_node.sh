#!/usr/bin/env bash
set -euo pipefail

# Download 10x Genomics Xenium Human Lymph Node Preview Data.
#
# Source:
# https://www.10xgenomics.com/datasets/human-lymph-node-preview-data-xenium-human-multi-tissue-and-cancer-panel-1-standard
#
# Usage:
#   bash download_10x_xenium_human_lymph_node.sh [output_dir]
#
# Check links and remote sizes without downloading:
#   CHECK_ONLY=1 bash download_10x_xenium_human_lymph_node.sh

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dataset_dir="${DATASET_DIR:-${script_dir}}"
out_dir="${1:-${dataset_dir}/samples}"
check_only="${CHECK_ONLY:-0}"
manifest="${MANIFEST:-${dataset_dir}/manifest/raw_urls.tsv}"

command -v curl >/dev/null 2>&1 || {
    echo "ERROR: curl is required." >&2
    exit 1
}

if [[ ! -f "$manifest" ]]; then
    echo "ERROR: manifest not found: $manifest" >&2
    exit 1
fi

md5_cmd=()
if command -v md5sum >/dev/null 2>&1; then
    md5_cmd=(md5sum)
elif command -v md5 >/dev/null 2>&1; then
    md5_cmd=(md5 -q)
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
    local size

    size="$(
        curl -fsSIL --retry 3 --retry-delay 5 "$url" |
            tr -d '\r' |
            awk 'tolower($1) == "content-length:" {print $2}' |
            tail -n 1
    )"

    if [[ -n "$size" ]]; then
        printf '%s\n' "$size"
        return
    fi

    curl -fsSL -r 0-0 -D - -o /dev/null "$url" |
        tr -d '\r' |
        awk 'tolower($1) == "content-range:" {for (i = 1; i <= NF; i++) if ($i ~ /\//) {sub(".*/", "", $i); print $i}}' |
        tail -n 1
}

download_file() {
    local sample_id="$1"
    local data_type="$2"
    local title="$3"
    local filename="$4"
    local bytes="$5"
    local md5="$6"
    local url="$7"
    local target="${out_dir}/${sample_id}/${filename}"

    printf '%-28s %10s  %s\n' "$title" "$(human_bytes "$bytes")" "$filename"

    if [[ "$check_only" == "1" ]]; then
        local actual_bytes
        actual_bytes="$(remote_size "$url")"
        if [[ "$actual_bytes" != "$bytes" ]]; then
            echo "ERROR: expected ${bytes} bytes but remote reports ${actual_bytes:-missing} for ${filename}" >&2
            exit 1
        fi
        return
    fi

    mkdir -p "$(dirname "$target")"
    if [[ -f "$target" ]]; then
        local existing_bytes
        existing_bytes="$(wc -c < "$target" | tr -d ' ')"
        if [[ "$existing_bytes" == "$bytes" ]]; then
            echo "  already complete: $target"
        else
            curl -fL --retry 5 --retry-delay 10 --retry-connrefused -C - -o "$target" "$url"
        fi
    else
        curl -fL --retry 5 --retry-delay 10 --retry-connrefused -C - -o "$target" "$url"
    fi

    local local_bytes
    local_bytes="$(wc -c < "$target" | tr -d ' ')"
    if [[ "$local_bytes" != "$bytes" ]]; then
        echo "ERROR: downloaded ${local_bytes} bytes but expected ${bytes} bytes for ${target}" >&2
        exit 1
    fi

    if [[ "${#md5_cmd[@]}" -gt 0 ]]; then
        local actual_md5
        actual_md5="$("${md5_cmd[@]}" "$target" | awk '{print $1}')"
        if [[ "$actual_md5" != "$md5" ]]; then
            echo "ERROR: MD5 mismatch for ${target}: got ${actual_md5}, expected ${md5}" >&2
            exit 1
        fi
    else
        echo "WARNING: no md5sum/md5 command found; skipped checksum for ${target}" >&2
    fi
}

total_bytes=0
total_objects=0
while IFS=$'\t' read -r sample_id data_type title filename expected_bytes md5 url; do
    url="${url%$'\r'}"
    [[ -z "${sample_id:-}" || "$sample_id" == "SampleID" ]] && continue
    total_bytes=$((total_bytes + expected_bytes))
    total_objects=$((total_objects + 1))
done < "$manifest"

echo "Output directory: ${out_dir}"
if [[ "$check_only" == "1" ]]; then
    echo "Mode: check only"
else
    echo "Mode: download with size and MD5 verification"
fi
echo
echo "Expected total size: $(human_bytes "$total_bytes")"
echo

ok=0
while IFS=$'\t' read -r sample_id data_type title filename expected_bytes md5 url; do
    url="${url%$'\r'}"
    [[ -z "${sample_id:-}" || "$sample_id" == "SampleID" ]] && continue
    download_file "$sample_id" "$data_type" "$title" "$filename" "$expected_bytes" "$md5" "$url"
    ok=$((ok + 1))
done < "$manifest"

echo
if [[ "$check_only" == "1" ]]; then
    echo "Finished: all ${ok}/${total_objects} verified 10x Xenium download links checked."
else
    echo "Finished: all ${ok}/${total_objects} verified 10x Xenium files downloaded and checked."
fi
