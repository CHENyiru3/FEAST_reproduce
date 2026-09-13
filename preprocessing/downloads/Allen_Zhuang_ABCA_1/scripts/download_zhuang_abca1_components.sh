#!/usr/bin/env bash
set -euo pipefail

# Download Allen Brain Cell Atlas Zhuang-ABCA-1 component data from the public S3 bucket.
#
# Source page:
# https://alleninstitute.github.io/abc_atlas_access/descriptions/Zhuang-ABCA-1.html
#
# Usage:
#   bash download_zhuang_abca1_components.sh [output_dir]
#
# Link/size check without downloading:
#   CHECK_ONLY=1 bash download_zhuang_abca1_components.sh
#
# Optional:
#   MANIFEST=/path/to/raw_urls.tsv bash download_zhuang_abca1_components.sh

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dataset_dir="$(cd "${script_dir}/.." && pwd)"
out_dir="${1:-${dataset_dir}/components}"
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

download_object() {
    local component="$1"
    local key="$2"
    local expected_bytes="$3"
    local url="$4"
    local target="${out_dir}/${key}"
    local component_label="${component//_/ }"

    printf '%-20s %10s  %s\n' "$component_label" "$(human_bytes "$expected_bytes")" "$key"

    if [[ "$check_only" == "1" ]]; then
        local actual_bytes
        actual_bytes="$(
            curl -fsSIL --retry 3 --retry-delay 5 "$url" |
                tr -d '\r' |
                awk 'tolower($1) == "content-length:" {print $2}' |
                tail -n 1
        )"

        if [[ "$actual_bytes" != "$expected_bytes" ]]; then
            echo "ERROR: expected ${expected_bytes} bytes but remote reports ${actual_bytes:-missing} for ${key}" >&2
            exit 1
        fi
        return
    fi

    mkdir -p "$(dirname "$target")"
    if [[ -f "$target" ]]; then
        local existing_bytes
        existing_bytes="$(wc -c < "$target" | tr -d ' ')"
        if [[ "$existing_bytes" == "$expected_bytes" ]]; then
            echo "  already complete: $target"
            return
        fi
    fi

    curl -fL --retry 5 --retry-delay 10 --retry-connrefused -C - -o "$target" "$url"

    local local_bytes
    local_bytes="$(wc -c < "$target" | tr -d ' ')"
    if [[ "$local_bytes" != "$expected_bytes" ]]; then
        echo "ERROR: downloaded ${local_bytes} bytes but expected ${expected_bytes} bytes for ${target}" >&2
        exit 1
    fi
}

total_objects=0
total_bytes=0

while IFS=$'\t' read -r sample data_type component key expected_bytes url; do
    url="${url%$'\r'}"
    [[ -z "${sample:-}" || "$sample" == "SampleID" ]] && continue
    total_objects=$((total_objects + 1))
    total_bytes=$((total_bytes + expected_bytes))
done < "$manifest"

echo "Output directory: ${out_dir}"
if [[ "$check_only" == "1" ]]; then
    echo "Mode: check only"
else
    echo "Mode: download"
fi
echo
echo "Expected total size: $(human_bytes "$total_bytes")"
echo

ok=0
while IFS=$'\t' read -r sample data_type component key expected_bytes url; do
    url="${url%$'\r'}"
    [[ -z "${sample:-}" || "$sample" == "SampleID" ]] && continue
    download_object "$component" "$key" "$expected_bytes" "$url"
    ok=$((ok + 1))
done < "$manifest"

echo
if [[ "$check_only" == "1" ]]; then
    echo "Finished: all ${ok}/${total_objects} objects checked."
else
    echo "Finished: all ${ok}/${total_objects} objects downloaded and size-checked."
fi
