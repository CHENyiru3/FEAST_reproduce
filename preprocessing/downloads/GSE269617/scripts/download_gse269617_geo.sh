#!/usr/bin/env bash
set -euo pipefail

# Download GEO supplementary files for GSE269617.
#
# Source:
# https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE269617&format=file
#
# Usage from Raw/GSE269617:
#   bash scripts/download_gse269617_geo.sh [output_dir]
#
# Check remote file sizes without downloading:
#   CHECK_ONLY=1 bash scripts/download_gse269617_geo.sh components

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dataset_dir="$(cd "${script_dir}/.." && pwd)"
out_dir="${1:-${dataset_dir}/components}"
check_only="${CHECK_ONLY:-0}"
manifest="${MANIFEST:-${dataset_dir}/manifest/raw_urls.tsv}"

command -v curl >/dev/null 2>&1 || {
    echo "ERROR: curl is required." >&2
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

remote_size() {
    local url="$1"
    curl -fsSIL --retry 3 --retry-delay 5 "$url" |
        tr -d '\r' |
        awk 'tolower($1) == "content-length:" {print $2}' |
        tail -n 1
}

validate_key() {
    local key="$1"
    if [[ -z "$key" || "$key" = /* || "$key" == ".." || "$key" == ../* || "$key" == */../* || "$key" == */.. ]]; then
        echo "ERROR: unsafe manifest key: ${key}" >&2
        exit 1
    fi
}

echo "GEO accession: GSE269617"
echo "Manifest: ${manifest}"
echo "Output directory: ${out_dir}"
if [[ "$check_only" == "1" ]]; then
    echo "Mode: check only"
else
    echo "Mode: download with size verification"
fi
echo

processed=0
total_expected=0

while IFS=$'\t' read -r sample_id data_type url component key expected_bytes md5 file_id notes; do
    if [[ "$sample_id" == "SampleID" || -z "$sample_id" ]]; then
        continue
    fi
    validate_key "$key"
    if [[ ! "$expected_bytes" =~ ^[0-9]+$ ]]; then
        echo "ERROR: invalid expected_bytes for ${key}: ${expected_bytes}" >&2
        exit 1
    fi

    target="${out_dir}/${key}"
    processed=$((processed + 1))
    total_expected=$((total_expected + expected_bytes))

    printf '%-20s %10s  %s\n' "$key" "$(human_bytes "$expected_bytes")" "$url"

    if [[ "$check_only" == "1" ]]; then
        actual_bytes="$(remote_size "$url")"
        if [[ "$actual_bytes" != "$expected_bytes" ]]; then
            echo "ERROR: expected ${expected_bytes} bytes but remote reports ${actual_bytes:-missing} for ${key}" >&2
            exit 1
        fi
        continue
    fi

    mkdir -p "$(dirname "$target")"

    if [[ -f "$target" ]]; then
        existing_bytes="$(wc -c < "$target" | tr -d ' ')"
        if [[ "$existing_bytes" == "$expected_bytes" ]]; then
            continue
        fi
    fi

    curl -fL --retry 5 --retry-delay 10 --retry-connrefused -C - -o "$target" "$url"

    local_bytes="$(wc -c < "$target" | tr -d ' ')"
    if [[ "$local_bytes" != "$expected_bytes" ]]; then
        echo "ERROR: downloaded ${local_bytes} bytes but expected ${expected_bytes} bytes for ${target}" >&2
        exit 1
    fi
done < "$manifest"

echo
echo "Expected total size: ${total_expected} bytes ($(human_bytes "$total_expected"))."

if [[ "$check_only" == "1" ]]; then
    echo "Finished: checked ${processed} GEO supplementary files."
else
    (
        cd "$dataset_dir"
        find components -type f -printf '%P\t%s\n' | sort > manifest/downloaded_files.tsv
        find components -type f -print0 | sort -z | xargs -0 sha256sum > checksums/sha256.tsv
        sha256sum -c checksums/sha256.tsv --quiet
    )
    echo "Finished: downloaded ${processed} GEO supplementary files."
    echo "Manifest: ${dataset_dir}/manifest/downloaded_files.tsv"
    echo "Checksums: ${dataset_dir}/checksums/sha256.tsv"
fi
