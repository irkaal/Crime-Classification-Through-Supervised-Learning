
# Data cleaning wrapper
preProcess <- function(dataset) {
  source('misc.R')
  source('outlier_detection.R')
  source('geospatial.R')
  loadPackages(c('data.table', 'reticulate'))
  source_python('dataCleaning.py')
  
  # Handle Outliers
  dataset <- handleOutliers(dataset)
  # Encode geospatial features
  dataset <- encodeGeospatial(dataset)
  # Main cleaning task (dataCleaning.py)
  dataset <- mainClean(dataset)
  
  dataset <- as.data.table(dataset)
  return(dataset)
}


# For use with Python rpy2. 
# Excluding mainClean() to prevent circular dependencies.
preProcessR <- function(dataset) {
  source('misc.R')
  source('outlier_detection.R')
  source('geospatial.R')
  loadPackages('data.table')
  dataset <- as.data.table(dataset)
  
  # Handle Outliers
  dataset <- handleOutliers(dataset)
  # Encode geospatial features
  dataset <- encodeGeospatial(dataset)
  
  dataset <- as.data.frame(dataset)
  return(dataset)
}
