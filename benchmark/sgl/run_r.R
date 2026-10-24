# R's SGL or sparsegl on a problem of problems.jl, at one mixture alpha, on the
# grid written there: the time of the whole path (median of REPS runs) and the
# solutions at every level. Both fit (1/(2n))||y - X b||^2 + lambda (alpha ||b||_1
# + (1 - alpha) sum_g sqrt(|g|) ||b_g||_2), without intercept or standardization.
#
#   Rscript benchmark/sgl/run_r.R PROBDIR ALPHA OUTDIR PKG [REPS]
#   PKG: SGL, SGL-tight (thresh 1e-6), sparsegl

args <- commandArgs(trailingOnly = TRUE)
d <- args[1]; alpha_str <- args[2]; out <- args[3]; pkg <- args[4]
reps <- if (length(args) > 4) as.integer(args[5]) else 3
alpha <- as.numeric(alpha_str)

meta <- strsplit(readLines(file.path(d, "meta.txt"))[1], " ")[[1]]
kv <- setNames(as.integer(sub(".*=", "", meta)), sub("=.*", "", meta))
n <- kv[["n"]]; p <- kv[["p"]]; gsize <- kv[["gsize"]]
X <- matrix(readBin(file.path(d, "X.bin"), "double", n * p), n, p)
y <- readBin(file.path(d, "y.bin"), "double", n)
group <- as.integer(readBin(file.path(d, "group.bin"), "double", p))
lfile <- file.path(d, sprintf("lambda_%s.bin", alpha_str))
lambda <- readBin(lfile, "double", file.size(lfile) / 8)

fit_once <- switch(pkg,
  "SGL" = function() {
    fit <- SGL::SGL(list(x = X, y = y), group, type = "linear", alpha = alpha,
                    standardize = FALSE, lambdas = lambda)
    fit$beta
  },
  "SGL-tight" = function() {
    fit <- SGL::SGL(list(x = X, y = y), group, type = "linear", alpha = alpha,
                    standardize = FALSE, lambdas = lambda, thresh = 1e-6, maxit = 10000)
    fit$beta
  },
  "sparsegl" = function() {
    fit <- sparsegl::sparsegl(X, y, group = group, family = "gaussian", lambda = lambda,
                              asparse = alpha, intercept = FALSE, standardize = FALSE)
    as.matrix(fit$beta)
  },
  stop("unknown package ", pkg))

times <- numeric(reps)
for (r in seq_len(reps)) {
  gc()
  times[r] <- system.time(B <- fit_once())[["elapsed"]]
}
dir.create(out, showWarnings = FALSE, recursive = TRUE)
stopifnot(nrow(B) == p, ncol(B) == length(lambda))
writeBin(as.double(B), file.path(out, sprintf("beta_%s.bin", pkg)))
writeLines(c(paste(times, collapse = " "), sprintf("version=%s", packageVersion(if (pkg == "sparsegl") "sparsegl" else "SGL"))),
           file.path(out, sprintf("time_%s.txt", pkg)))
cat(sprintf("%s alpha=%s %s: median %.3f s over %d runs\n", basename(d), alpha_str, pkg, median(times), reps))
