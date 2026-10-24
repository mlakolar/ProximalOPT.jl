# One reference path, called by cross_language.jl INSIDE a Slurm allocation.
# No package installation is performed here. Install QUIC and glasso in the
# run's R library before execution, and retain installation failures separately.
#
# Interfaces checked against the CRAN sources:
# https://github.com/cran/QUIC/blob/1235758fe364d275565f11d2ca96276d7eadc6ba/R/QUIC.R
# https://github.com/cran/glasso/blob/master/R/glasso.R
# https://cran.r-project.org/web/packages/glasso/glasso.pdf
# QUIC's X and glasso's wi are precision matrices. Both receive the full
# off-diagonal penalty matrix; no factor of n or 2 rescales the objective.

args <- commandArgs(trailingOnly = TRUE)
arg <- function(name, default = NULL) {
  k <- match(name, args)
  if (is.na(k)) return(default)
  if (k == length(args)) stop(paste(name, "needs a value"))
  args[[k + 1L]]
}
root <- normalizePath(arg("--root"), mustWork = TRUE)
p <- as.integer(arg("--p"))
round <- as.integer(arg("--round"))
solver <- arg("--solver")
quic_tol <- as.numeric(arg("--reference-tol", "1e-4"))
glasso_thr <- as.numeric(arg("--glasso-thr", "1e-6"))
max_iter <- as.integer(arg("--max-iter", "100"))
fit_limit <- as.numeric(arg("--fit-limit", "120"))
stopifnot(p >= 2L, round >= 1L, solver %in% c("R_QUIC", "R_glasso"))
Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1")
set.seed(2200L + p + round)
lambda <- scan(file.path(root, "lambda.txt"), quiet = TRUE)
prefix <- sprintf("p%d_r%d_%s", p, round, solver)
output <- file.path(root, paste0(prefix, ".tsv"))
con <- file(output, "wt")
writeLines("p\tround\tlevel\tlambda\tsolver\tseconds\titerations\tstatus\ttheta_file\tdetail", con)
clean <- function(s) gsub("[\t\r\n]", " ", as.character(s))
row <- function(k, elapsed, iterations, status, theta_file = "", detail = "") {
  writeLines(paste(p, round, k, format(lambda[k], digits = 17), solver,
                   format(elapsed, digits = 17), iterations, status,
                   theta_file, clean(detail), sep = "\t"), con)
  flush(con)
}
package <- if (solver == "R_QUIC") "QUIC" else "glasso"
if (!requireNamespace(package, quietly = TRUE)) {
  for (k in seq_along(lambda)) row(k, NA_real_, NA_integer_, "unavailable", detail = paste("package", package, "not installed"))
  close(con)
  quit(status = 0L)
}
message("R ", getRversion(), " package ", package, " ", packageVersion(package),
        " p=", p, " round=", round)
raw_S <- readBin(file.path(root, sprintf("S_p%d.bin", p)), what = "double",
                 n = p * p, size = 8L, endian = "little")
stopifnot(length(raw_S) == p * p, all(is.finite(raw_S)))
S <- matrix(raw_S, nrow = p, ncol = p)
previous <- NULL
for (k in seq_along(lambda)) {
  rho <- matrix(lambda[k], p, p)
  diag(rho) <- 0
  fit <- NULL
  failure <- NULL
  gc()
  # setTimeLimit is a soft guard: compiled foreign code may check only on
  # return. The Julia parent also bounds this whole R child process.
  setTimeLimit(cpu = Inf, elapsed = fit_limit, transient = TRUE)
  elapsed <- system.time(tryCatch({
    if (solver == "R_QUIC") {
      fit <- if (is.null(previous)) {
        QUIC::QUIC(S, rho, tol = quic_tol, msg = 0L, maxIter = max_iter)
      } else {
        QUIC::QUIC(S, rho, tol = quic_tol, msg = 0L, maxIter = max_iter,
                   X.init = previous$X, W.init = previous$W)
      }
    } else {
      fit <- glasso::glasso(S, rho, thr = glasso_thr, maxit = max_iter,
                            approx = FALSE, penalize.diagonal = FALSE,
                            start = if (is.null(previous)) "cold" else "warm",
                            w.init = if (is.null(previous)) NULL else previous$w,
                            wi.init = if (is.null(previous)) NULL else previous$wi,
                            trace = FALSE)
    }
  }, error = function(e) failure <<- conditionMessage(e)))[["elapsed"]]
  setTimeLimit(cpu = Inf, elapsed = Inf, transient = FALSE)
  if (!is.null(failure)) {
    row(k, elapsed, NA_integer_, "error", detail = failure)
    next
  }
  theta <- matrix(if (solver == "R_QUIC") fit$X else fit$wi, p, p)
  iterations <- if (solver == "R_QUIC") as.integer(fit$iter[1L]) else as.integer(fit$niter)
  if (!all(is.finite(theta))) {
    row(k, elapsed, iterations, "error", detail = "nonfinite precision matrix")
    next
  }
  theta_file <- sprintf("theta_%s_k%d.bin", prefix, k)
  writeBin(as.double(theta), file.path(root, theta_file), size = 8L, endian = "little")
  row(k, elapsed, iterations, "returned", theta_file,
      paste("package", packageVersion(package), "warm", !is.null(previous)))
  previous <- fit
}
close(con)
