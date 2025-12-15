library(dplyr)
library(sf)
library(readr)
library(tools)

TIMESTAMP_FORMAT <- "%Y-%m-%d %H:%M:%S"

folder <- "data_folder"
files <- list.files(folder, pattern = "\\.csv$", full.names=TRUE)

for (file in files) {
  
  message("Processing: ", file)

  tryCatch({
    # Basic processing:
    geojson <- read_csv(file, show_col_types=FALSE) %>%
      mutate(
        track_timestamp = as.POSIXct(track_timestamp, format=TIMESTAMP_FORMAT)
      ) %>%
      filter(
        !is.na(track_longitude),
        !is.na(track_latitude),
        !is.na(track_altitude)
      )
    
    # Creating .geojson file:
    geojson_sf <- st_as_sf(
      geojson %>%
        select(identification_number, track_timestamp, track_heading, track_speed,
               track_vertical_speed, track_longitude, track_latitude, track_altitude),
      coords = c("track_longitude", "track_latitude", "track_altitude"),
      crs = 4326,
      remove = FALSE
    )
    
    # Building trajectories:
    routes <- geojson_sf %>%
      group_by(identification_number) %>%
      arrange(track_timestamp, .by_group=TRUE) %>%
      summarise(geometry = st_combine(geometry)) %>%
      st_cast("LINESTRING")
    
    # Attach metadata:
    routes <- routes %>%
      left_join(
        geojson %>%
          select(identification_number, Airline, airport_origin_icao) %>%
          distinct(),
        by = "identification_number"
      )
    
    # Write the file:
    out_file <- file.path(
      folder, paste0(file_path_sans_ext(basename(file)), "_processed.geojson")
    )
    
    st_write(routes, out_file, driver = "GeoJSON", append = FALSE)
    
  }, error = function(e) {
    message("Error in ", file, ": ", e$message)
  })
}

