

# TODO: visualize outliers
detectOutliers <- function(dataset) {
  source('misc.R')
  loadPackages('ggplot2')
  
  plot <- ggplot(data = dataset, aes(group = PdDistrict, color = PdDistrict))
  
  plot + geom_boxplot(aes(y = X)) + ggtitle('Distribution of Longitudes by District')
  
  plot + geom_boxplot(aes(y = Y)) + ggtitle('Distribution of Latitudes by District')
  
  model <- kmeans(dataset[, c('X', 'Y')], 5)
  dataset$Cluster <- factor(model$cluster)
  ggplot(data = dataset, aes(x = X, y = Y)) + geom_point(data = dataset, aes(color = Cluster))
}


handleOutliers <- function(dataset) {
  source('misc.R')
  loadPackages(c('stringr', 'tmaptools'))
  
  # Define outlier boundaries. These are just estimates from google map.
  min_X <- -122.515465
  max_X <- -122.356443
  min_Y <- 37.707462
  max_Y <- 37.834977
  filter_X <- (min_X < dataset$X) & (dataset$X > max_X)
  filter_Y <- (min_Y < dataset$Y) & (dataset$Y > max_Y)
  outlier_filter <- filter_X | filter_Y
  outlier <- dataset[outlier_filter, ]
  
  # Return if no outliers found
  if (!nrow(outlier)) return(dataset)
  
  #
  # 1st Pass (Use matching address coordinates)
  #
  
  # Check if there is a non-outlier with the same address
  match_filter <- dataset$Address %in% outlier$Address & !outlier_filter
  match <- dataset[match_filter, ]
  
  # Sort the data.table by Address to allow factor indexing trick to work
  match <- match[order(Address)]
  
  # Replace the incorrect coordinates using factor index
  replace_filter <- dataset$Address %in% match$Address & outlier_filter
  replace_index <- as.integer(factor(dataset[replace_filter, ]$Address))
  dataset[replace_filter, ]$X <- match$X[replace_index]
  dataset[replace_filter, ]$Y <- match$Y[replace_index]
  
  
  #
  # 2nd Pass (Use tmap API and OpenStreetMap Nominatim to geocode locations of the outliers)
  #
  
  # Update outliers
  filter_X <- (min_X < dataset$X) & (dataset$X > max_X)
  filter_Y <- (min_Y < dataset$Y) & (dataset$Y > max_Y)
  outlier_filter <- filter_X | filter_Y
  outlier <- dataset[outlier_filter, ]
  
  # DO NOT RUN THIS REPEATEDLY! USE CACHED RESULTS INSTEAD.
  # Policy (https://operations.osmfoundation.org/policies/nominatim/)
  # query <- str_replace(outlier$Address, '/', 'and')
  # geocode <- geocode_OSM(query)
  # write.csv(geocode, './data/geocode_result.csv', row.names = F)
  path <- unzip('./data/sf-crime.zip', 'geocode_result.csv')
  geocode <- fread(path); invisible(file.remove(path))
  valid_filter <- (min_X < geocode$lon & geocode$lon < max_X) | (min_Y < geocode$lat & geocode$lat < max_Y)
  geocode <- geocode[valid_filter, ]
  geocode_address <- str_replace(geocode$query, 'and', '/')
  
  # Replace the incorrect coordinates using the valid geocode query result
  replace_filter <- dataset$Address %in% geocode_address
  dataset[replace_filter, ]$X <- geocode$lon
  dataset[replace_filter, ]$Y <- geocode$lat
  
  #
  # 3rd Pass (Estimate)
  #
  
  # Update outliers
  filter_X <- (min_X < dataset$X) & (dataset$X > max_X)
  filter_Y <- (min_Y < dataset$Y) & (dataset$Y > max_Y)
  outlier_filter <- filter_X | filter_Y
  outlier <- dataset[outlier_filter, ]
  
  # Get mean coordinates by PdDistrict
  non_outlier <- dataset[!outlier_filter, ]
  district_list <- list(PdDistrict = non_outlier$PdDistrict)
  avg_X <- aggregate(non_outlier[, 'X'], district_list, mean)
  avg_Y <- aggregate(non_outlier[, 'Y'], district_list, mean)
  
  # Replace the incorrect coordinates 
  replace_index <- as.numeric(factor(outlier$PdDistrict))
  dataset[outlier_filter, ]$X <- avg_X[replace_index, ]$X
  dataset[outlier_filter, ]$Y <- avg_Y[replace_index, ]$Y  

  return(dataset)
}

