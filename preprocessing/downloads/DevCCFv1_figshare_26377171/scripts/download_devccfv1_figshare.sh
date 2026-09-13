#!/usr/bin/env bash
set -euo pipefail

# Download Figshare dataset: DevCCFv1
#
# Source:
# https://figshare.com/articles/dataset/DevCCFv1/26377171
#
# Usage from Raw/DevCCFv1_figshare_26377171:
#   bash scripts/download_devccfv1_figshare.sh [output_dir]
#
# Check Figshare metadata and links without downloading files:
#   CHECK_ONLY=1 bash scripts/download_devccfv1_figshare.sh components

article_id="${FIGSHARE_ARTICLE_ID:-26377171}"
dataset_key="DevCCFv1"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dataset_dir="$(cd "${script_dir}/.." && pwd)"
out_dir="${1:-${dataset_dir}/components}"
check_only="${CHECK_ONLY:-0}"
manifest="${MANIFEST:-${dataset_dir}/manifest/raw_urls.tsv}"
metadata_json="${dataset_dir}/manifest/figshare_article_${article_id}_metadata.json"
api_url="https://api.figshare.com/v2/articles/${article_id}"

command -v curl >/dev/null 2>&1 || {
    echo "ERROR: curl is required." >&2
    exit 1
}

command -v jq >/dev/null 2>&1 || {
    echo "ERROR: jq is required." >&2
    exit 1
}

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

is_placeholder_manifest() {
    [[ ! -s "$manifest" ]] && return 0
    local lines
    lines="$(wc -l < "$manifest" | tr -d ' ')"
    [[ "$lines" -le 1 ]]
}

refresh_manifest() {
    local metadata_tmp
    local manifest_tmp
    metadata_tmp="$(mktemp)"
    manifest_tmp="$(mktemp)"

    curl -fsSL --retry 3 --retry-delay 5 "$api_url" -o "$metadata_tmp"

    mkdir -p "${dataset_dir}/manifest"
    cp "$metadata_tmp" "$metadata_json"

    printf 'SampleID\tdata_type\turl\tcomponent\tkey\texpected_bytes\tmd5\tfile_id\tnotes\n' > "$manifest_tmp"
    jq -r --arg dataset_key "$dataset_key" --arg article_id "$article_id" '
        . as $article
        | .files[]
        | . as $file
        | ($article.folder_structure[($file.id | tostring)] // "root") as $folder
        | ($folder | gsub("^/+"; "") | gsub("/+$"; "")) as $clean_folder
        | ($file.name | gsub("^/+"; "")) as $clean_name
        | (if $clean_folder == "" or $clean_folder == "root" then $clean_name else "\($clean_folder)/\($clean_name)" end) as $key
        | [
            $dataset_key,
            "figshare_file",
            $file.download_url,
            (if $clean_folder == "" then "root" else $clean_folder end),
            $key,
            ($file.size | tostring),
            ($file.computed_md5 // $file.supplied_md5 // "-"),
            ($file.id | tostring),
            ("Figshare article " + $article_id)
          ]
        | @tsv
    ' "$metadata_tmp" >> "$manifest_tmp"

    mv "$manifest_tmp" "$manifest"
    rm -f "$metadata_tmp"
}

check_url() {
    local url="$1"
    local expected_bytes="$2"
    local headers
    local actual_bytes

    if headers="$(curl -fsSIL --retry 3 --retry-delay 5 "$url" 2>/dev/null | tr -d '\r')"; then
        actual_bytes="$(
            printf '%s\n' "$headers" |
                awk 'tolower($1) == "content-length:" {print $2}' |
                tail -n 1
        )"
        if [[ -n "$actual_bytes" && "$actual_bytes" != "$expected_bytes" ]]; then
            echo "ERROR: expected ${expected_bytes} bytes but remote reports ${actual_bytes} for ${url}" >&2
            exit 1
        fi
    else
        curl -fsSL --retry 3 --retry-delay 5 --range 0-0 -o /dev/null "$url"
    fi
}

validate_key() {
    local key="$1"
    if [[ -z "$key" || "$key" = /* || "$key" == ".." || "$key" == ../* || "$key" == */../* || "$key" == */.. ]]; then
        echo "ERROR: unsafe manifest key: ${key}" >&2
        exit 1
    fi
}

verify_file() {
    local target="$1"
    local expected_bytes="$2"
    local expected_md5="$3"
    local actual_bytes

    actual_bytes="$(wc -c < "$target" | tr -d ' ')"
    if [[ "$actual_bytes" != "$expected_bytes" ]]; then
        echo "ERROR: ${target} has ${actual_bytes} bytes; expected ${expected_bytes}" >&2
        exit 1
    fi

    if [[ "$expected_md5" != "-" && "${#md5_cmd[@]}" -gt 0 ]]; then
        local actual_md5
        actual_md5="$("${md5_cmd[@]}" "$target" | awk '{print $1}')"
        if [[ "$actual_md5" != "$expected_md5" ]]; then
            echo "ERROR: MD5 mismatch for ${target}: got ${actual_md5}, expected ${expected_md5}" >&2
            exit 1
        fi
    elif [[ "$expected_md5" != "-" ]]; then
        echo "WARNING: no md5sum/md5 command found; skipped checksum for ${target}" >&2
    fi
}

if [[ "${REFRESH_MANIFEST:-0}" == "1" ]] || is_placeholder_manifest || [[ ! -s "$metadata_json" ]]; then
    refresh_manifest
fi

title="$(jq -r '.title' "$metadata_json")"
doi="$(jq -r '.doi' "$metadata_json")"
version="$(jq -r '.version' "$metadata_json")"
file_count="$(jq -r '.files | length' "$metadata_json")"
total_bytes="$(jq -r '.size' "$metadata_json")"

echo "Article ID: ${article_id}"
echo "Title: ${title}"
echo "DOI: ${doi}"
echo "Version: ${version}"
echo "Files: ${file_count}"
echo "Expected total size: $(human_bytes "$total_bytes")"
echo "Manifest: ${manifest}"
echo "Output directory: ${out_dir}"
if [[ "$check_only" == "1" ]]; then
    echo "Mode: check only"
else
    echo "Mode: download with size and MD5 verification"
fi
echo

processed=0
total_manifest_bytes=0

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
    total_manifest_bytes=$((total_manifest_bytes + expected_bytes))
    processed=$((processed + 1))

    printf '%-14s %10s  %s\n' "$file_id" "$(human_bytes "$expected_bytes")" "$key"

    if [[ "$check_only" == "1" ]]; then
        check_url "$url" "$expected_bytes"
        continue
    fi

    mkdir -p "$(dirname "$target")"

    if [[ -f "$target" ]]; then
        existing_bytes="$(wc -c < "$target" | tr -d ' ')"
        if [[ "$existing_bytes" == "$expected_bytes" ]]; then
            verify_file "$target" "$expected_bytes" "$md5"
            continue
        fi
    fi

    curl -fL --retry 5 --retry-delay 10 --retry-connrefused -C - -o "$target" "$url"
    verify_file "$target" "$expected_bytes" "$md5"
done < "$manifest"

echo
if [[ "$check_only" == "1" ]]; then
    echo "Finished: checked ${processed} Figshare files."
    echo "Manifest total: ${total_manifest_bytes} bytes ($(human_bytes "$total_manifest_bytes"))."
else
    (
        cd "$dataset_dir"
        find components -type f -printf '%P\t%s\n' | sort > manifest/downloaded_files.tsv
        find components -type f -print0 | sort -z | xargs -0 sha256sum > checksums/sha256.tsv
        sha256sum -c checksums/sha256.tsv --quiet
    )
    echo "Finished: downloaded ${processed} Figshare files."
    echo "Manifest: ${dataset_dir}/manifest/downloaded_files.tsv"
    echo "Checksums: ${dataset_dir}/checksums/sha256.tsv"
fi
