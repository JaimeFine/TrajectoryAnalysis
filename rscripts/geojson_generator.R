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
        file, show_col_types=FALSE
      ) %>%
      mutate(
        track_timestamp = as.POSIXct(track_timestamp, format=TIMESTAMP_FORMAT),
        track_longitude = as.numeric(track_longitude),
        track_latitude = as.numeric(track_latitude),
        track_altitude = as.numeric(track_altitude),
        track_speed = as.numeric(track_speed),
        track_heading = as.numeric(track_heading),
        track_vertical_speed = as.numeric(track_verical_speed)
      ) %>%
      filter(
        !is.na(track_longitude),
        !is.na(track_latitude),
        !is.na(track_altitude),
        !is.na(track_timestamp)
      )
    
    # Adding new elements:
    geojson <- geojson %>%
      arrange(identification_number, track_timestamp) %>%
      group_by(identification_number) %>%
      mutate(
        dt = as.numeric(
          difftime(
            lead(track_timestamp),
            track_timestamp,
            units = "secs"
          )
        )
      ) %>%
      mutate(
        vx = track_speed * sin(track_heading * pi / 180),
        vy = track_speed * cos(track_heading * pi / 180),
        vz = track_vertical_speed
      )
    
    geojson <- geojson %>%
      mutate(
        state = purrr::pmap(
          list(
            track_longitude,
            track_latitude,
            track_altitude,
            vx, vy, vz
          ),
          function(lon, lat, alt, vx, vy, vz) {
            list(
              pos = c(lon, lat, alt),
              vel = c(vx, vy, vz)
            )
          }
        )
      )
    
    # Creating .geojson file:
    geojson_sf <- st_as_sf(
      geojson,
      coords = c("track_longitude", "track_latitude", "track_altitude"),
      crs = 4326,
      remove = FALSE
    ) %>%
      transmute(
        geometry,
        properties = purrr::pmap(
          list(
            state,
            track_timestamp,
            dt,
            identification_number,
            Airline,
            airport_origin_icao
          ),
          function(state, time, dt, id, airline, origin) {
            list(
              state = state,
              timestamp = as.character(time),
              dt = dt,
              flight_id = id,
              Airline = airline,
              airport_origin_icao = origin
            )
          }
        )
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

