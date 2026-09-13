#!/usr/bin/env bash
set -euo pipefail

# Download raw data links from:
# https://research.libd.org/spatialLIBD/ -> Access the data -> Raw data
#
# Usage:
#   bash download_spatialLIBD_raw_data.sh [output_dir]
#
# Link check without downloading files:
#   CHECK_ONLY=1 bash download_spatialLIBD_raw_data.sh
#
# Optional:
#   MANIFEST=/path/to/raw_urls.tsv bash download_spatialLIBD_raw_data.sh

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dataset_dir="$(cd "${script_dir}/.." && pwd)"
out_dir="${1:-${dataset_dir}/samples}"
manifest="${MANIFEST:-${dataset_dir}/manifest/raw_urls.tsv}"
check_only="${CHECK_ONLY:-0}"

if [[ ! -f "$manifest" ]]; then
    echo "ERROR: manifest not found: $manifest" >&2
    exit 1
fi

if command -v curl >/dev/null 2>&1; then
    downloader="curl"
elif command -v wget >/dev/null 2>&1; then
    downloader="wget"
else
    echo "ERROR: install curl or wget first." >&2
    exit 1
fi

download_url() {
    local url="$1"
    local target="$2"

    if [[ "$check_only" == "1" ]]; then
        if [[ "$downloader" == "curl" ]]; then
            curl -fsSIL --retry 3 --retry-delay 5 "$url" >/dev/null
        else
            wget --spider -q "$url"
        fi
        return
    fi

    mkdir -p "$(dirname "$target")"
    if [[ "$downloader" == "curl" ]]; then
        curl -fsSL --retry 5 --retry-delay 10 --retry-connrefused -C - -o "$target" "$url"
    else
        wget -c -O "$target" "$url"
    fi
}

total=0
ok=0

while IFS=$'\t' read -r sample data_type url; do
    url="${url%$'\r'}"
    [[ -z "${sample:-}" || "$sample" == "SampleID" ]] && continue
    total=$((total + 1))

    filename="$(basename "$url")"
    target="${out_dir}/${sample}/${filename}"

    if [[ "$check_only" == "1" ]]; then
        printf '[check] %s %-12s %s\n' "$sample" "$data_type" "$url"
    else
        printf '[download] %s %-12s %s\n' "$sample" "$data_type" "$target"
    fi

    download_url "$url" "$target"
    ok=$((ok + 1))
done < "$manifest"

printf 'Finished: %s/%s links %s.\n' "$ok" "$total" "$([[ "$check_only" == "1" ]] && echo checked || echo downloaded)"
