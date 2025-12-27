from collections import defaultdict
import numpy as np
from datetime import datetime
import json
# from structure_lib import Node, LinkedList

# -------------------- Block 1 ----------------- # 
#      Preprocessing with the flight data        #
# ---------------------------------------------- #

flights = defaultdict(lambda: {
    "coords": [],
    "vel": [],
    # "time": [],
    "dt": []
})

with open("D:/ADataBase/flights_data_geojson/2024-11-10/2024-11-10-CAN_processed.geojson") as raw:
    geojson = json.load(raw)

for feature in geojson["features"]:
    props = feature["properties"]
    geom = feature["geometry"]
    
    f_id = props["flight_id"]
    dt = props["dt"]

    lon, lat, alt = geom["coordinates"]
    vx, vy, vz = map(float, props["velocity"].split())

    # timestamp = datetime.fromisoformat(props["timestamp"])
    
    flights[f_id]["coords"].append([lon, lat, alt])
    flights[f_id]["vel"].append([vx, vy, vz])
    flights[f_id]["dt"].append(dt)
    # flights[f_id]["time"].append(timestamp)

for f_id in flights:
    flights[f_id]["coords"] = np.array(flights[f_id]["coords"])
    flights[f_id]["vel"] = np.array(flights[f_id]["vel"])
    flights[f_id]["dt"] = np.array(flights[f_id]["dt"])
    
# ------------------ Block 2 ----------------- # 
#            Computation heavy zone            #
# -------------------------------------------- #
"""
from julia import Julia
jl = Julia(compiled_modules=False)
from julia import LinearAlgebra
"""

# Pure physics-based model ----- basic:
physic_normal = {}
for f_id in flights:
    coords = flights[f_id]["coords"]
    vel = flights[f_id]["vel"]
    dt = flights[f_id]["dt"]

    size = len(coords) - 1
    physic_normal[f_id] = np.zeros((size, 3), dtype = float)

    for i in range(size):
        # The phsic_matrix's first column is the prediction for 2nd position!!!
        dx = coords[i] + vel[i+1] * dt[i+1]
        physic_normal[f_id][i] = dx
        
# Pure physics-based model ----- advanced:
better_matrix = np.array()

# Physics-ML model:


# Sparse attention?

