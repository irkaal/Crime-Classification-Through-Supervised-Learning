

loadPackages <- function(pkgs) {
  new_pkgs <- pkgs[!(pkgs %in% installed.packages()[ ,'Package'])]
  if(length(new_pkgs)) install.packages(new_pkgs)
  invisible(lapply(pkgs, library, character.only = T))
}

