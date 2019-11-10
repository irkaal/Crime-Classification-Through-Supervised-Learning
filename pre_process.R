

# Data cleaning wrapper
pre_process <- function(dataset) {
  source('misc.R')
  source('outlier_detection.R')
  source('geospatial.R')
  loadPackages(c('reticulate'))
  source_python('dataCleaning.py')

  # Handle Outliers
  dataset <- handleOutliers(dataset)
  
  # Encode geospatial features
  dataset <- encodeGeospatial(dataset)
  
  # Main cleaning task (dataCleaning.py)
  dataset <- mainClean(dataset)
  
  dataset <- data.table(dataset)
  return(dataset)
}

