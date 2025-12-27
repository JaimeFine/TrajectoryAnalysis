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
        dx = coords[i] + vel[i] * dt[i]
        physic_normal[f_id][i] = dx

"""
Obviously, the output:
...
[ 3.93346485e+04, -9.70215903e+04,  3.61000000e+04],
[ 1.97690042e+04, -4.85948999e+04,  3.61000000e+04],
[ 3.93348216e+04, -9.70219568e+04,  3.61000000e+04],
[-5.31731053e+04, -7.33233705e+04,  3.61000000e+04],
[-3.95573973e+04, -1.84714838e+04,  3.99400000e+04],
[-3.95575754e+04, -1.84716463e+04,  3.61000000e+04],
[-3.42670276e+04, -1.60046179e+04,  3.61000000e+04],
[-1.35313544e+04, -6.92548368e+03,  3.61000000e+04],
...
is stupid!
"""

# Pure physics-based model ----- advanced:
from scipy.interpolate import CubicHermiteSpline

flight_alpha = {}

for f_id in flights:
    coords = flights[f_id]["coords"]
    vel = flights[f_id]["vel"]
    dt = flights[f_id]["dt"]

    

physic_better = {}
for f_id in flights:
    coords = flights[f_id]["coords"]
    vel = flights[f_id]["vel"]
    dt = flights[f_id]["dt"]
    size = len(coords)

    curvatures = []
    a = np.zeros((size, 3))

    for i in range(1, size-1):
        a[i] = (vel[i+1] - vel[i-1]) / (dt[i-1] + dt[i])

        speed = np.linalg.norm(vel[i])
        if speed > 1e-7:
            k = np.linalg.norm(np.cross(vel[i], a)) / speed**3
            curvatures.append(k)

    curvatures = np.array(curvatures)
    k95 = np.percentile(curvatures, 95)

    flight_alpha[f_id] = np.log(5) / k95

    physic_better[f_id] = np.zeros((size, 3), dtype = float)

    for i in range(2, size-2):
        idx = [i-2, i-1, i+1, i+2]

        t0 = 0.0
        t1 = dt[i-2]
        tm = dt[i-2] + dt[i-1]
        t2 = dt[i-2] + dt[i-1] + dt[i]
        t3 = dt[i-2] + dt[i-1] + dt[i] + dt[i+1]

        t = np.array([t0, t1, t2, t3])
        p = coords[idx]
        v = vel[idx]

        spline_x = CubicHermiteSpline(t, p[:,0], v[:,0])
        spline_y = CubicHermiteSpline(t, p[:,1], v[:,1])
        spline_z = CubicHermiteSpline(t, p[:,2], v[:,2])

        # Get the Spline prediction:
        spline = np.array([
            spline_x(tm), spline_y(tm), spline_z(tm)
        ], dtype=float)

        # Calculate the Constant-Acceleration prediction:
        ca = coords[i-1] + vel[i-1] * dt[i-1] + a[i]/2 * dt[i-1]**2

        # Local curvature:
        speed = np.linalg.norm(vel[i])
        if speed < 1e-7 or speed > 1e+7:
            k = 0.0
        else:
            k = np.linalg.norm(np.cross(vel[i], a)) / (speed**3)

        w = np.exp(-50 * k)
        pred = w * ca + (1 - w) * spline

        physic_better[f_id][i] = pred

# Physics-ML model:


# Sparse attention?

