library(dplyr)
library(sf)
library(readr)
library(tools)

TIMESTAMP_FORMAT <- "%Y-%m-%d %H:%M:%S"

folder <- "C:/Users/"
files <- list.files(folder, pattern = "\\.csv$", full.names=TRUE)

for (file in files) {
  
  message("Processing: ", file)

  tryCatch({
    # Basic processing:
    geojson <- read_csv(
        file, show_col_types=FALSE, col_types = cols(.default = col_character())
      ) %>%
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
      vcoords = c("track_speed, track_heading")
      crs = 4326,
      remove = FALSE
    )
    
    # Attach metadata:
    geojson_sf <- geojson_sf %>%
      rename(
        spd = track_speed,
        vspd = track_vertical_speed,
        hd = track_heading,
        time = track_timestamp
      ) %>%
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
    
    st_write(geojson_sf, out_file, driver = "GeoJSON", append = FALSE)
    
  }, error = function(e) {
    message("Error in ", file, ": ", e$message)
  })
}

