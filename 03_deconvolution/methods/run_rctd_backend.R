#!/usr/bin/env Rscript
#
# RCTD deconvolution backend using Bioconductor spacexr 1.2.0.
#
# Uses SummarizedExperiment for reference, SpatialExperiment for spatial data,
# createRctd() and runRctd(rctd_mode = "full") per the Bioconductor API.
#
# Usage:
#   Rscript run_rctd_backend.R \
#     --counts <counts.csv> \
#     --coords <coords.csv> \
#     --reference <reference.csv> \
#     --output <output.csv> \
#     [--cache-dir /tmp/rctd_bioc_cache]

suppressPackageStartupMessages({
  library(spacexr)
  library(SummarizedExperiment)
  library(SpatialExperiment)
  library(Matrix)
})

# ---- Parse CLI ----
args <- commandArgs(trailingOnly = TRUE)

parse_arg <- function(name, args) {
  idx <- which(args == name)
  if (length(idx) == 0) stop(paste("Missing required argument:", name))
  args[idx + 1]
}

counts_file  <- parse_arg("--counts", args)
coords_file  <- parse_arg("--coords", args)
ref_file     <- parse_arg("--reference", args)
ct_file      <- parse_arg("--cell-types", args)
output_file  <- parse_arg("--output", args)
cache_dir    <- if ("--cache-dir" %in% args) {
                  parse_arg("--cache-dir", args)
                } else {
                  "/tmp/rctd_bioc_cache"
                }

# ---- Load spatial data ----
cat("Loading spatial count matrix...\n")
spatial_counts <- as.matrix(read.csv(counts_file, row.names = 1, check.names = FALSE))
spatial_counts <- Matrix(spatial_counts, sparse = TRUE)
cat(sprintf("  %d genes x %d spots\n", nrow(spatial_counts), ncol(spatial_counts)))

cat("Loading coordinates...\n")
coords <- read.csv(coords_file, row.names = 1)
nUMI_spatial <- as.integer(round(coords$nUMI))
names(nUMI_spatial) <- rownames(coords)
spatial_coords <- as.matrix(coords[, c("x", "y")])
colnames(spatial_coords) <- c("x", "y")
rownames(spatial_coords) <- colnames(spatial_counts)

# ---- Load reference data (full single-cell format) ----
cat("Loading reference...\n")
ref_df <- read.csv(ref_file, row.names = 1, check.names = FALSE)
ref_mat <- as.matrix(ref_df)
storage.mode(ref_mat) <- "integer"
cat(sprintf("  %d genes x %d cells\n", nrow(ref_mat), ncol(ref_mat)))

cat("Loading cell type labels...\n")
ref_cell_types <- read.csv(ct_file, row.names = 1, check.names = FALSE)
ct_labels <- factor(ref_cell_types$cell_type)
names(ct_labels) <- rownames(ref_cell_types)
cat(sprintf("  %d unique cell types\n", length(levels(ct_labels))))

# ---- Build SummarizedExperiment reference ----
# Ensure nUMI >= 100 to bypass spacexr's internal UMI filtering on reference cells
nUMI_ref <- pmax(as.integer(round(colSums(ref_mat))), 100L)
names(nUMI_ref) <- colnames(ref_mat)

cat(sprintf("Building reference SummarizedExperiment (%d cells, %d cell types)...\n",
            ncol(ref_mat), length(levels(ct_labels))))
reference_se <- SummarizedExperiment(
  assays = list(counts = ref_mat),
  colData = data.frame(
    cell_type = ct_labels,
    nUMI = nUMI_ref,
    row.names = colnames(ref_mat)
  )
)

# ---- Build SpatialExperiment ----
cat("Building SpatialExperiment...\n")
spatial_spe <- SpatialExperiment(
  assays = list(counts = spatial_counts),
  spatialCoords = spatial_coords
)
colData(spatial_spe)$nUMI <- nUMI_spatial

# ---- Run RCTD in full mode ----
cat("Running createRctd...\n")
rctd_data <- createRctd(spatial_spe, reference_se,
                         UMI_min = 1, UMI_min_sigma = 1)

cat("Running runRctd (full mode)...\n")
results_spe <- runRctd(rctd_data, rctd_mode = "full", max_cores = 1)

# ---- Extract results ----
cat("Extracting results...\n")
stopifnot("weights" %in% assayNames(results_spe))

# weights assay: cell_types as rows, spots as columns (Bioconductor convention)
# Transpose to spots x cell_types for output CSV
wt <- as.matrix(assay(results_spe, "weights"))
wt <- t(wt)  # now spots x cell_types

# Normalize so each row sums to 1
row_sums <- rowSums(wt)
row_sums[row_sums == 0] <- 1
wt <- wt / row_sums
wt[is.na(wt)] <- 0

# ---- Write output ----
cat(sprintf("Writing proportions (%d spots x %d cell types) to %s\n",
            nrow(wt), ncol(wt), output_file))
out_df <- as.data.frame(wt)
rownames(out_df) <- paste0("spot_", seq_len(nrow(out_df)) - 1)

dir.create(dirname(output_file), showWarnings = FALSE, recursive = TRUE)
write.csv(out_df, file = output_file, row.names = TRUE)

cat("Done.\n")
