
# Data cleaning wrapper
preProcess <- function(dataset, centerScale = F, rlang = T) {
  source('misc.R'); loadPackages(c('data.table', 'reticulate'))
  
  if (rlang) {
    dataset <- handleOutliers(dataset)
    source_python('data_cleaning.py')
    dataset <- as.data.table(main_clean(dataset, centerScale))
    return(dataset)
  } else { # If called from python
    handleOutliers(as.data.table(dataset), write = T)
    return(NULL)
  }
}

# Replaces the outlier coordinates with the correct ones or uses the mean coordinates of their respective PdDistrict
handleOutliers <- function(dataset, write = F, runGeocode = F) {
  source('misc.R'); loadPackages('stringr')
  tic <- start('Handling Outliers...', 10)
  
  updateProgress(1, 10, 'Identifying outliers')
  # Define outlier boundaries. These are just estimates from google map.
  min_X <- -122.515465
  max_X <- -122.356443
  min_Y <- 37.707462
  max_Y <- 37.834977
  filter_X <- (min_X < dataset$X) & (dataset$X > max_X)
  filter_Y <- (min_Y < dataset$Y) & (dataset$Y > max_Y)
  outlier_filter <- filter_X | filter_Y
  outlier <- dataset[outlier_filter, ]
  
  # Return if no outliers or address column found
  if (!nrow(outlier) || !('Address' %in% colnames(dataset))) {
    end(tic, 10)
    return(dataset)
  }
  
  # 1st Pass (Use matching address coordinates)
  
  # Check if there is a non-outlier with the same address
  updateProgress(2, 10, 'Checking for matching address')
  match_filter <- dataset$Address %in% outlier$Address & !outlier_filter
  match <- dataset[match_filter, ]
  
  # Sort the data.table by Address to allow factor indexing trick to work
  match <- match[order(Address)]
  
  # Replace the incorrect coordinates using factor index
  updateProgress(3, 10, 'Replacing coordinates')
  replace_filter <- dataset$Address %in% match$Address & outlier_filter
  replace_index <- as.integer(factor(dataset[replace_filter, ]$Address))
  dataset[replace_filter, ]$X <- match$X[replace_index]
  dataset[replace_filter, ]$Y <- match$Y[replace_index]
  
  # 2nd Pass (Use tmap API and OpenStreetMap Nominatim to geocode locations of the outliers)
  
  # Update outliers
  updateProgress(4, 10, 'Updating outliers')
  filter_X <- (min_X < dataset$X) & (dataset$X > max_X)
  filter_Y <- (min_Y < dataset$Y) & (dataset$Y > max_Y)
  outlier_filter <- filter_X | filter_Y
  outlier <- dataset[outlier_filter, ]
  
  # DO NOT RUN THIS REPEATEDLY! USE SAVED RESULT INSTEAD.
  # Policy (https://operations.osmfoundation.org/policies/nominatim/)
  updateProgress(5, 10, 'Geocoding')
  if (runGeocode) {
    loadPackages('tmaptools')
    query <- str_replace(outlier$Address, '/', 'and')
    geocode <- geocode_OSM(query)
  } else { # Use saved result
    path <- unzip('data/sf-crime.zip', 'geocode_result.csv')
    geocode <- fread(path); invisible(file.remove(path))
    valid_filter <- (min_X < geocode$lon & geocode$lon < max_X) | (min_Y < geocode$lat & geocode$lat < max_Y)
    geocode <- geocode[valid_filter, ]
    geocode_address <- str_replace(geocode$query, 'and', '/')  
  }
  
  # Replace the incorrect coordinates using the valid geocode query result
  updateProgress(6, 10, 'Replacing coordinates')
  replace_filter <- dataset$Address %in% geocode_address
  dataset[replace_filter, ]$X <- geocode$lon
  dataset[replace_filter, ]$Y <- geocode$lat
  
  # 3rd Pass (Estimate the remaining outlier coordinates)
  
  # Update outliers
  updateProgress(7, 10, 'Updating outliers')
  filter_X <- (min_X < dataset$X) & (dataset$X > max_X)
  filter_Y <- (min_Y < dataset$Y) & (dataset$Y > max_Y)
  outlier_filter <- filter_X | filter_Y
  outlier <- dataset[outlier_filter, ]
  
  # Get mean coordinates by PdDistrict
  updateProgress(8, 10, 'Estimating coordinates by PdDistrict')
  non_outlier <- dataset[!outlier_filter, ]
  district_list <- list(PdDistrict = non_outlier$PdDistrict)
  avg_X <- aggregate(non_outlier[, 'X'], district_list, mean)
  avg_Y <- aggregate(non_outlier[, 'Y'], district_list, mean)
  
  # Replace the incorrect coordinates 
  updateProgress(9, 10, 'Replacing coordinates')
  replace_index <- as.numeric(factor(outlier$PdDistrict))
  dataset[outlier_filter, ]$X <- avg_X[replace_index, ]$X
  dataset[outlier_filter, ]$Y <- avg_Y[replace_index, ]$Y  
  
  if (write) {
    fwrite(dataset, 'data/train_rclean.csv', row.names = F) 
    end(tic, 10)
    return(NULL)
  } else {
    end(tic, 10)
    return(dataset)
  }
}
