#!/usr/bin/env bash
set -euo pipefail

# Download Tencent SpatialOmics datasetID=119.
#
# Source page:
# https://gene.ai.tencent.com/SpatialOmics/dataset?datasetID=119
#
# Dataset metadata endpoint:
# https://gene.ai.tencent.com/SpatialOmics/api/dataset/119
#
# Usage:
#   bash download_tencent_spatialomics_dataset119.sh [output_dir]
#
# Check links and remote sizes without downloading:
#   CHECK_ONLY=1 bash download_tencent_spatialomics_dataset119.sh

dataset_id="${DATASET_ID:-119}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dataset_dir="$(cd "${script_dir}/.." && pwd)"
out_dir="${1:-${dataset_dir}/samples}"
check_only="${CHECK_ONLY:-0}"
api_base="https://gene.ai.tencent.com/SpatialOmics/api"
manifest="${dataset_dir}/manifest/raw_urls.tsv"
metadata_json="${dataset_dir}/manifest/dataset_${dataset_id}_metadata.json"

command -v curl >/dev/null 2>&1 || {
    echo "ERROR: curl is required." >&2
    exit 1
}

command -v jq >/dev/null 2>&1 || {
    echo "ERROR: jq is required." >&2
    exit 1
}

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

safe_path_part() {
    printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_'
}

metadata_tmp="$(mktemp)"
manifest_tmp="$(mktemp)"
trap 'rm -f "$metadata_tmp" "$manifest_tmp"' EXIT

curl -fsSL --retry 3 --retry-delay 5 "${api_base}/dataset/${dataset_id}" -o "$metadata_tmp"

code="$(jq -r '.code' "$metadata_tmp")"
if [[ "$code" != "0" ]]; then
    echo "ERROR: metadata API returned code=${code}" >&2
    jq -r '.msg // empty' "$metadata_tmp" >&2
    exit 1
fi

dataset_name="$(jq -r '.data.name_short' "$metadata_tmp")"
dataset_access="$(jq -r '.data.access' "$metadata_tmp")"
dataset_title="$(jq -r '.data.name_long' "$metadata_tmp")"
sample_count="$(jq -r '.data.data | length' "$metadata_tmp")"

mkdir -p "${dataset_dir}/manifest"
cp "$metadata_tmp" "$metadata_json"

printf 'SampleID\tdata_type\tdata_id\tsample_name\tfilename\texpected_bytes\turl\n' > "$manifest_tmp"

echo "Dataset ID: ${dataset_id}"
echo "Dataset: ${dataset_name} (${dataset_access})"
echo "Title: ${dataset_title}"
echo "Samples: ${sample_count}"
echo "Output directory: ${out_dir}"
if [[ "$check_only" == "1" ]]; then
    echo "Mode: check only"
else
    echo "Mode: download"
fi
echo

total_remote_bytes=0
checked=0

while IFS=$'\t' read -r data_id sample_name; do
    url="${api_base}/download_data/${data_id}"
    headers="$(
        curl -fsSIL --retry 3 --retry-delay 5 "$url" |
            tr -d '\r'
    )"

    bytes="$(
        printf '%s\n' "$headers" |
            awk 'tolower($1) == "content-length:" {print $2}' |
            tail -n 1
    )"
    filename="$(
        printf '%s\n' "$headers" |
            sed -n -E 's/.*filename="?([^";]+)"?.*/\1/p' |
            tail -n 1
    )"

    if [[ -z "$bytes" || ! "$bytes" =~ ^[0-9]+$ ]]; then
        echo "ERROR: could not determine remote size for dataID=${data_id}" >&2
        exit 1
    fi

    if [[ -z "$filename" ]]; then
        filename="${dataset_name}_${sample_name}_${data_id}.h5ad"
    fi

    total_remote_bytes=$((total_remote_bytes + bytes))
    checked=$((checked + 1))

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$sample_name" "h5ad" "$data_id" "$sample_name" "$filename" "$bytes" "$url" >> "$manifest_tmp"

    printf '%-8s %-24s %10s  %s\n' "$data_id" "$sample_name" "$(human_bytes "$bytes")" "$filename"

    if [[ "$check_only" == "1" ]]; then
        continue
    fi

    sample_dir="$(safe_path_part "$sample_name")"
    target="${out_dir}/${sample_dir}/${filename}"
    mkdir -p "$(dirname "$target")"

    if [[ -f "$target" ]]; then
        local_bytes="$(wc -c < "$target" | tr -d ' ')"
        if [[ "$local_bytes" == "$bytes" ]]; then
            echo "  already complete: $target"
            continue
        fi
    fi

    curl -fL --retry 5 --retry-delay 10 --retry-connrefused -C - -o "$target" "$url"

    local_bytes="$(wc -c < "$target" | tr -d ' ')"
    if [[ "$local_bytes" != "$bytes" ]]; then
        echo "ERROR: downloaded ${local_bytes} bytes but expected ${bytes} bytes for ${target}" >&2
        exit 1
    fi
done < <(jq -r '.data.data[] | [.id, .name] | @tsv' "$metadata_tmp")

mv "$manifest_tmp" "$manifest"

echo
if [[ "$check_only" == "1" ]]; then
    echo "Finished: checked ${checked}/${sample_count} data files."
    echo "Remote total: ${total_remote_bytes} bytes ($(human_bytes "$total_remote_bytes"))."
    echo "Manifest: ${manifest}"
else
    echo "Finished: downloaded ${checked}/${sample_count} data files."
    echo "Downloaded total: ${total_remote_bytes} bytes ($(human_bytes "$total_remote_bytes"))."
    echo "Manifest: ${manifest}"
fi
