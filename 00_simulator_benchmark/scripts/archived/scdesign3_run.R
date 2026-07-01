#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(optparse)
    library(Matrix)
    library(SingleCellExperiment)
    library(scDesign3)
})

option_list <- list(
    make_option(c("-c", "--cellinfo"), type = "character", default = NULL),
    make_option(c("-g", "--geneinfo"), type = "character", default = NULL),
    make_option(c("-m", "--matrix"), type = "character", default = NULL),
    make_option(c("-s", "--spatial_loc"), type = "character", default = NULL),
    make_option(c("-o", "--output_dir"), type = "character", default = "./"),
    make_option(c("--seed"), type = "integer", default = 2026),
    make_option(c("--n_cores"), type = "integer", default = 2),
    make_option(c("--family_use"), type = "character", default = "nb"),
    make_option(c("--copula"), type = "character", default = "gaussian"),
    make_option(c("--important_feature"), type = "character", default = "auto"),
    make_option(c("--parallelization"), type = "character", default = "mcmapply"),
    make_option(c("--mu_formula"), type = "character",
                default = "spatial1 + spatial2 + offset(log(library))"),
    make_option(c("--sigma_formula"), type = "character", default = "1"),
    make_option(c("--corr_formula"), type = "character", default = "1"),
    make_option(c("--usebam"), action = "store_true", default = FALSE),
    make_option(c("--edf_flexible"), action = "store_true", default = FALSE),
    make_option(c("--trace"), action = "store_true", default = FALSE)
)

opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$cellinfo) || is.null(opt$geneinfo) ||
    is.null(opt$matrix) || is.null(opt$spatial_loc)) {
    stop("All input files must be specified")
}

dir.create(opt$output_dir, showWarnings = FALSE, recursive = TRUE)

write_metadata <- function(path, metadata) {
    if (requireNamespace("jsonlite", quietly = TRUE)) {
        writeLines(jsonlite::toJSON(metadata, auto_unbox = TRUE, pretty = TRUE), path)
    } else {
        capture.output(str(metadata), file = path)
    }
}

read_id_column <- function(frame, candidates, fallback_index = TRUE) {
    for (column in candidates) {
        if (column %in% colnames(frame)) {
            return(as.character(frame[[column]]))
        }
    }
    if (fallback_index) {
        return(as.character(seq_len(nrow(frame))))
    }
    NULL
}

resolve_label_column <- function(cellinfo) {
    candidates <- c("ground_truth", "cell_type", "annotation", "region", "cluster", "label")
    for (column in candidates) {
        if (column %in% colnames(cellinfo)) {
            values <- as.character(cellinfo[[column]])
            values <- values[!is.na(values)]
            if (length(unique(values)) > 1) {
                return(column)
            }
        }
    }
    NULL
}

important_feature_value <- function(value) {
    if (identical(value, "auto")) {
        return(0.95)
    }
    if (identical(value, "all")) {
        return("all")
    }
    numeric_value <- suppressWarnings(as.numeric(value))
    if (is.na(numeric_value)) {
        stop("important_feature must be auto, all, or a numeric value")
    }
    numeric_value
}

status_path <- file.path(opt$output_dir, "scdesign3_metadata.json")
start_time <- Sys.time()

tryCatch({
    set.seed(opt$seed)

    cellinfo <- read.csv(opt$cellinfo, check.names = FALSE)
    geneinfo <- read.csv(opt$geneinfo, check.names = FALSE)
    spatial <- read.csv(opt$spatial_loc, check.names = FALSE)
    counts <- as(readMM(opt$matrix), "dgCMatrix")

    spot_ids <- read_id_column(cellinfo, c("spot_id", "Cell", "cell", "barcode", "label"))
    gene_ids <- read_id_column(geneinfo, c("gene_id", "gene", "genes", "gene_name", "Gene"))

    if (nrow(counts) == length(spot_ids) && ncol(counts) == length(gene_ids)) {
        counts <- t(counts)
    }
    if (nrow(counts) != length(gene_ids) || ncol(counts) != length(spot_ids)) {
        stop(
            "Matrix shape does not match gene and spot tables: matrix=",
            paste(dim(counts), collapse = "x"),
            ", genes=", length(gene_ids),
            ", spots=", length(spot_ids)
        )
    }

    rownames(counts) <- make.unique(gene_ids)
    colnames(counts) <- make.unique(spot_ids)

    if (!all(c("x", "y") %in% colnames(spatial))) {
        numeric_cols <- names(spatial)[vapply(spatial, is.numeric, logical(1))]
        if (length(numeric_cols) < 2) {
            stop("spatial.csv must contain x/y or at least two numeric columns")
        }
        spatial$x <- spatial[[numeric_cols[1]]]
        spatial$y <- spatial[[numeric_cols[2]]]
    }
    if (nrow(spatial) != ncol(counts)) {
        stop("spatial row count does not match count columns")
    }

    coldata <- DataFrame(cellinfo)
    rownames(coldata) <- colnames(counts)
    coldata$spatial1 <- as.numeric(spatial$x)
    coldata$spatial2 <- as.numeric(spatial$y)
    coldata$library <- pmax(Matrix::colSums(counts), 1)

    label_column <- resolve_label_column(cellinfo)
    if (!is.null(label_column)) {
        coldata$scdesign3_celltype <- factor(as.character(cellinfo[[label_column]]))
        celltype_arg <- "scdesign3_celltype"
    } else {
        celltype_arg <- NULL
    }

    sce <- SingleCellExperiment(list(counts = counts), colData = coldata)
    feature_arg <- important_feature_value(opt$important_feature)

    cat("scDesign3: fitting and simulating", nrow(counts), "genes x",
        ncol(counts), "spots\n")
    cat("scDesign3: label column:",
        ifelse(is.null(label_column), "none", label_column), "\n")

    simu <- scDesign3::scdesign3(
        sce = sce,
        assay_use = "counts",
        celltype = celltype_arg,
        pseudotime = NULL,
        spatial = c("spatial1", "spatial2"),
        other_covariates = "library",
        ncell = ncol(sce),
        mu_formula = opt$mu_formula,
        sigma_formula = opt$sigma_formula,
        family_use = opt$family_use,
        n_cores = opt$n_cores,
        usebam = opt$usebam,
        edf_flexible = opt$edf_flexible,
        corr_formula = opt$corr_formula,
        copula = opt$copula,
        DT = TRUE,
        pseudo_obs = FALSE,
        important_feature = feature_arg,
        nonnegative = TRUE,
        nonzerovar = FALSE,
        return_model = FALSE,
        parallelization = opt$parallelization,
        trace = opt$trace
    )

    sim_counts <- simu$new_count
    if (is.null(sim_counts)) {
        stop("scDesign3 returned NULL new_count")
    }
    sim_counts <- as(sim_counts, "dgCMatrix")
    if (nrow(sim_counts) == ncol(counts) && ncol(sim_counts) == nrow(counts)) {
        sim_counts <- t(sim_counts)
    }
    if (nrow(sim_counts) != nrow(counts) || ncol(sim_counts) != ncol(counts)) {
        stop(
            "Simulated count shape mismatch: simulated=",
            paste(dim(sim_counts), collapse = "x"),
            ", expected=", paste(dim(counts), collapse = "x")
        )
    }

    sim_counts@x[!is.finite(sim_counts@x)] <- 0
    sim_counts@x[sim_counts@x < 0] <- 0
    sim_counts@x <- round(sim_counts@x)
    rownames(sim_counts) <- rownames(counts)
    colnames(sim_counts) <- colnames(counts)

    loc <- data.frame(
        x = as.numeric(spatial$x),
        y = as.numeric(spatial$y),
        label = colnames(counts),
        check.names = FALSE
    )

    writeMM(sim_counts, file.path(opt$output_dir, "scdesign3_count.mtx"))
    write.csv(
        data.frame(gene = rownames(sim_counts), check.names = FALSE),
        file.path(opt$output_dir, "scdesign3_genes.csv"),
        row.names = FALSE
    )
    write.csv(loc, file.path(opt$output_dir, "scdesign3_loc.csv"), row.names = FALSE)

    metadata <- list(
        tool = "scDesign3",
        status = "success",
        seed = opt$seed,
        n_genes = nrow(counts),
        n_spots = ncol(counts),
        family_use = opt$family_use,
        mu_formula = opt$mu_formula,
        sigma_formula = opt$sigma_formula,
        corr_formula = opt$corr_formula,
        copula = opt$copula,
        n_cores = opt$n_cores,
        usebam = opt$usebam,
        edf_flexible = opt$edf_flexible,
        parallelization = opt$parallelization,
        important_feature = opt$important_feature,
        resolved_label_column = ifelse(is.null(label_column), NA, label_column),
        started_at = as.character(start_time),
        ended_at = as.character(Sys.time()),
        elapsed_seconds = as.numeric(difftime(Sys.time(), start_time, units = "secs")),
        package_version = as.character(utils::packageVersion("scDesign3"))
    )
    write_metadata(status_path, metadata)
    cat("scDesign3: completed successfully\n")
}, error = function(e) {
    metadata <- list(
        tool = "scDesign3",
        status = "failed",
        error = conditionMessage(e),
        seed = opt$seed,
        started_at = as.character(start_time),
        ended_at = as.character(Sys.time()),
        elapsed_seconds = as.numeric(difftime(Sys.time(), start_time, units = "secs")),
        family_use = opt$family_use,
        mu_formula = opt$mu_formula,
        sigma_formula = opt$sigma_formula,
        corr_formula = opt$corr_formula,
        copula = opt$copula
    )
    write_metadata(status_path, metadata)
    cat("scDesign3 error:\n")
    cat(conditionMessage(e), "\n")
    quit(status = 1)
})
