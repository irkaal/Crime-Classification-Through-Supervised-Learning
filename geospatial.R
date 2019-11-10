

encodeGeospatial <- function(dataset) {
  source('misc.R')
  loadPackages('dplyr')

  dataset <- encodePatrolDiv(dataset)
  dataset <- encodeHaverDist(dataset)
  return(dataset)  
}


encodePatrolDiv <- function(dataset) {
  if ('PdDistrict' %in% colnames(dataset)) {
    # PatrolDivision (https://en.wikipedia.org/wiki/San_Francisco_Police_Department)
    # Patrol divisions are broken down into two divisions Golden Gate Division and Metro Division which are each led by San Francisco Police Commanders.
    dataset <- dataset %>% 
      mutate(
        MetroDiv = as.integer(PdDistrict %in% c('CENTRAL', 'INGLESIDE', 'NORTHERN', 'SOUTHERN', 'TENDERLOIN')),
        GoldenGateDiv = as.integer(PdDistrict %in% c('BAYVIEW', 'MISSION', 'PARK', 'RICHMOND', 'TARAVAL'))) 
  }
  return(dataset)
}


encodeHaverDist <- function(dataset) {
  if (Reduce('&', c('X', 'Y') %in% colnames(dataset))) {
    # Add distances between center of the city + all police stations and crime locations.
    dataset <- dataset %>% 
      mutate(
        # Center of city (https://www.citylab.com/design/2016/06/exact-center-of-san-francisco/486341/)
        CenterDist  = haversine(X, Y, -122.442500, 37.754540), 
        # Coordinates obtained from google maps
        CentralDist = haversine(X, Y, -122.409960, 37.798736), 
        IngleDist   = haversine(X, Y, -122.446261, 37.724694), 
        NorthDist   = haversine(X, Y, -122.432516, 37.780226),
        SouthDist   = haversine(X, Y, -122.389411, 37.772382),
        TenderDist  = haversine(X, Y, -122.412924, 37.783783),
        BayDist     = haversine(X, Y, -122.397771, 37.729825),
        MissionDist = haversine(X, Y, -122.421951, 37.763013),
        ParkDist    = haversine(X, Y, -122.455391, 37.767835),
        RichDist    = haversine(X, Y, -122.464462, 37.780016),
        TaravalDist = haversine(X, Y, -122.481516, 37.743755))
  }
  return(dataset)
}


####################
# Helper functions #
####################


# Conversions
radians <- function(deg) {
  rad <- deg * pi / 180
  return(rad)
}
degrees <- function(rad) {
  deg <- rad * 180 / pi
  return(deg)
}


# Haversine distance - Great-circle distance (Kilometres)
haversine <- function(X1, Y1, X2, Y2) {
  X1 <- radians(X1)
  Y1 <- radians(Y1)
  X2 <- radians(X2)
  Y2 <- radians(Y2)
  r <- 6378.137 # Earth's radius in metres
  h <- sin((Y2 - Y1) / 2)^2 + cos(Y1) * cos(Y2) * sin((X2 - X1) / 2)^2
  d <- 2 * r * asin(sqrt(h))
  return(d)
}


# Manhattan distance L1-norm
manhattan <- function(X1, Y1, X2, Y2) {
  lat_dist <- haversine(X1, Y1, X1, Y2)
  lon_dist <- haversine(X1, Y1, X2, Y1)
  man_dist <- lat_dist + lon_dist
  return(man_dist)
}


# Bearing degree
bearing <- function(X1, Y1, X2, Y2) {
  X1 <- radians(X1)
  Y1 <- radians(Y1)
  X2 <- radians(X2)
  Y2 <- radians(Y2)
  y <- sin(X2 - X1) * cos(Y2)
  x <- cos(Y1) * sin(Y2) - sin(Y1) * cos(Y2) * cos(X2 - X1)
  theta <- degrees(atan2(y, x))
  return(theta)
}


# Polar coordinates
polarCoord <- function(X, Y) {
  return(list(
    Rho = sqrt(X^2 + Y^2),
    Phi = atan(Y / X)
  ))
}

