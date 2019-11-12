
loadPackages <- function(pkgs, quietly = F) {
  doWork <- function() {
    new_pkgs <- pkgs[!(pkgs %in% installed.packages()[ ,'Package'])]
    if(length(new_pkgs)) install.packages(new_pkgs)
    invisible(lapply(pkgs, library, character.only = T))
  }
  if (quietly) suppressPackageStartupMessages(doWork()) else doWork()
}

sourceFiles <- function(scripts) invisible(lapply(scripts, source))

'%nin%' <- Negate('%in%')

# Progress helpers

updateProgress <- function(i, max_i, task) {
  progress <- paste(rep('=', 48 / max_i * i), collapse = '')
  space <- paste(rep(' ', 36 - nchar(task)), collapse = '')
  cat(sprintf('\r[%-48s] %d%% (%s)%s', progress, 100 / max_i * i, task, space))
}

start <- function(task, max_i) {
  cat('R>', task, '\n')
  updateProgress(0, max_i, '')
  return(Sys.time())
}

end <- function(tic, max_i) {
  updateProgress(max_i, max_i, 'Done')
  cat('\nElapsed time:', round(as.numeric(Sys.time() - tic), 3), 'second(s)\n')
}
