#!/usr/bin/env bash
set -euo pipefail

# Download the GSE251926 metastatic lymph node 3D supplementary h5ad file.
#
# Usage:
#   bash scripts/download_GSE251926_metastatic_lymph_node_3d.sh [output_dir]
#
# Link check without downloading:
#   CHECK_ONLY=1 bash scripts/download_GSE251926_metastatic_lymph_node_3d.sh
#
# Optional:
#   MANIFEST=/path/to/raw_urls.tsv bash scripts/download_GSE251926_metastatic_lymph_node_3d.sh

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dataset_dir="$(cd "${script_dir}/.." && pwd)"
out_dir="${1:-${dataset_dir}/samples}"
manifest="${MANIFEST:-${dataset_dir}/manifest/raw_urls.tsv}"
check_only="${CHECK_ONLY:-0}"
log_dir="${dataset_dir}/logs"
log_file="${log_dir}/download_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$log_dir"
exec > >(tee -a "$log_file") 2>&1

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

filename_from_geo_url() {
    local url="$1"
    python3 - "$url" <<'PY'
import os
import sys
from urllib.parse import parse_qs, unquote, urlparse

url = sys.argv[1]
parsed = urlparse(url)
query = parse_qs(parsed.query)
if query.get("file"):
    print(unquote(query["file"][0]))
else:
    print(os.path.basename(parsed.path))
PY
}

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

    filename="$(filename_from_geo_url "$url")"
    target="${out_dir}/${sample}/${filename}"

    if [[ "$check_only" == "1" ]]; then
        printf '[check] %s %-12s %s\n' "$sample" "$data_type" "$url"
    else
        printf '[download] %s %-12s %s\n' "$sample" "$data_type" "$target"
    fi

    download_url "$url" "$target"
    ok=$((ok + 1))
done < "$manifest"

printf 'Finished: %s/%s links %s. Log: %s\n' "$ok" "$total" "$([[ "$check_only" == "1" ]] && echo checked || echo downloaded)" "$log_file"
