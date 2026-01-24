import numpy as np
import pandas as pd
import faiss
import json
import time
from scipy.spatial import cKDTree
from infomap import Infomap

# --- 1. COORDINATE CONSTANTS & FUNCTIONS ---
axis = 6378137.0
flattening = 1 / 298.257223563
eccentricity2 = flattening * (2 - flattening)

def geodetic2ecef(lon, lat, hei):
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    N = axis / np.sqrt(1 - eccentricity2 * np.sin(lat)**2)
    x = (N + hei) * np.cos(lat) * np.cos(lon)
    y = (N + hei) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - eccentricity2) + hei) * np.sin(lat)
    return np.array([x, y, z])

def adf(x, index, pos, s, k=100, sigma0=500.0):
    _, idx = index.search(x.reshape(1, 3), k)
    neighbors = pos[idx[0]]
    scores = s[idx[0]]
    diff = neighbors - x
    sigma = sigma0 / (scores + 1e-6)
    quadratic = np.sum(diff ** 2 * (1.0 / (sigma ** 2))[:, None], axis=1)
    return np.sum(scores * np.exp(-0.5 * quadratic))

# --- 2. DATA PREPARATION ---
df = pd.read_csv("D:/ADataBase/china_poi.csv")
pos = np.vstack([geodetic2ecef(lo, la, al) for lo, la, al in df[["lon", "lat", "alt"]].values]).astype("float32")
s = df["poi_score"].to_numpy()

quantizer = faiss.IndexFlatL2(3)
index = faiss.IndexIVFFlat(quantizer, 3, 4096)
index.train(pos)
index.add(pos)
index.nprobe = 16

# --- 3. TRAJECTORY PROCESSING ---
with open("D:/ADataBase/flights_data_geojson/2024-12-16/2024-12-16-CTU_processed.geojson") as f:
    track = json.load(f)

track_coords = np.array([feat["geometry"]["coordinates"] for feat in track["features"] if feat["geometry"]["type"] == "Point"])
# Pre-convert track to ECEF for distance calculations
track_ecef = np.vstack([geodetic2ecef(p[0], p[1], p[2]) for p in track_coords]).astype("float32")

print("Calculating ADF for trajectory...")
track_adf = np.array([adf(p, index, pos, s) for p in track_ecef])

# --- 4. OPTIMIZED SENSITIVITY SWEEP ---
alphas_to_test = [1.1, 1.2, 1.3, 1.4, 1.5]
all_results = []
sigma_graph = 500.0 # Standardized metric sigma (meters)

for a in alphas_to_test:
    t_start = time.time()
    baseline = np.median(track_adf)
    is_zoi = track_adf >= (a * baseline)
    zoi_indices = np.where(is_zoi)[0]
    
    communities_map = {}
    valid_comm_count = 0

    if len(zoi_indices) > 0:
        # Use ECEF (meters) for accurate graph distances
        zoi_pos = track_ecef[zoi_indices]
        zoi_adf_vals = track_adf[zoi_indices]
        
        tree = cKDTree(zoi_pos)
        pairs = np.array(list(tree.query_pairs(r=sigma_graph * 5)))
        
        im = Infomap("--two-level --silent")
        
        if len(pairs) > 0:
            # VECTORIZED EDGE CALCULATION (No NetworkX)
            i, j = pairs[:, 0], pairs[:, 1]
            # Euclidean distance in meters
            dists = np.linalg.norm(zoi_pos[i] - zoi_pos[j], axis=1)
            # Gaussian weights
            weights = np.minimum(zoi_adf_vals[i], zoi_adf_vals[j]) * np.exp(-dists / sigma_graph)
            
            # Mask out zero weights and add to Infomap directly
            mask = weights > 1e-6
            i_f, j_f, w_f = i[mask], j[mask], weights[mask]
            
            for idx in range(len(i_f)):
                im.add_link(int(i_f[idx]), int(j_f[idx]), float(w_f[idx]))
        
        im.run()
        communities_map = {node.node_id: node.module_id for node in im.nodes}
        valid_comm_count = len(set(communities_map.values()))

    # Build results list using list comprehension (faster than per-index loops)
    for i, idx in enumerate(range(len(track_coords))):
        # Local mapping: zoi_indices stores the original track index
        # We need to see if idx exists in zoi_indices to find its community
        comm_id = -1
        if is_zoi[idx]:
            # Find the position of this index in the zoi_indices array
            local_id = np.where(zoi_indices == idx)[0][0]
            comm_id = communities_map.get(local_id, -1)

        all_results.append({
            "lon": track_coords[idx, 0],
            "lat": track_coords[idx, 1],
            "ADF": track_adf[idx],
            "alpha_val": a,
            "is_zoi": int(is_zoi[idx]),
            "community_id": comm_id
        })
    
    print(f"Alpha {a:.1f} | Time: {time.time()-t_start:.2f}s | Communities: {valid_comm_count}")

pd.DataFrame(all_results).to_csv("alpha_sensitivity_results.csv", index=False)