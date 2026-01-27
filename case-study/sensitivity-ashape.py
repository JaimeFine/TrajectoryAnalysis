import pandas as pd
import numpy as np
from infomap import Infomap
from KDEpy import FFTKDE
from skimage import measure
import geopandas as gpd
from scipy.spatial import cKDTree
import time

t_start = time.perf_counter()

print(f"[{time.perf_counter()-t_start:.2f}s] Loading data...")
df = pd.read_csv("C:/Users/13647/OneDrive/Desktop/MiMundo/Projects/TrajectoryAnalysis/data/trajectory_adf_zoi.csv")

df = df[df["ZOI"] == 1]

gdf_points = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.lon, df.lat),
    crs="EPSG:4326"
)

gdf_points = gdf_points.to_crs(epsg=32648)
coords_m = np.vstack([gdf_points.geometry.x.values, gdf_points.geometry.y.values]).T
adf_values = df['ADF'].values

t0 = time.perf_counter()
print(f"[{time.perf_counter()-t_start:.2f}s] Building KDTree...")
tree = cKDTree(coords_m)

sigma_m = 1000.0
max_dist = sigma_m * 5
pairs = tree.query_pairs(r=max_dist)
print(f"[{time.perf_counter()-t_start:.2f}s] Found {len(pairs)} potential edges.")

log_interval = 100
count = 0
total_pairs = len(pairs)
chunk_start = time.perf_counter()

if pairs:
    pair_array = np.array(list(pairs))
    i, j = pair_array.T
    
    dist = np.linalg.norm(coords_m[i] - coords_m[j], axis=1)
    weight = min(adf_values[i], adf_values[j]) * np.exp(-dist / sigma_m)
    
    mask = weight > 0
    edge_list = list(zip(i[mask], j[mask], weight[mask]))
    
    if count % log_interval == 0 or count == total_pairs:
        now = time.perf_counter()
        elapsed = now - chunk_start
        speed = log_interval / elapsed if elapsed > 0 else 0
        percent = (count / total_pairs) * 100
        print(f"{percent:>6.1f}% | {count:>10} edges | speed: {speed:>6.0f}/s")
        chunk_start = time.perf_counter()
else:
    edge_list = []

print(f"Graph construction complete in {time.perf_counter() - t0:.2f}s")

# --- RUN INFOMAP ---
print(f"[{time.perf_counter()-t_start:.2f}s] Running Infomap...")
infomap_wrapper = Infomap("--two-level --silent")
for u, v, w in edge_list:
    infomap_wrapper.add_link(u, v, w)
infomap_wrapper.run()

communities = [node.module_id for node in infomap_wrapper.nodes]
unique_comms, counts = np.unique(communities, return_counts=True)
    
valid_communities = sum(1 for count in counts if count >= 4)

for comm_id in unique_comms:
    mask = (np.array(communities) == comm_id)
    X = coords_m[mask]
    weights = adf_values[mask]

    kde = FFTKDE(bw=500).fit(X, weights=weights) # same to the ADF sigma
    grid, density = kde.evaluate(grid_points=1024)
    _, point_density = kde.evaluate(X)

    rho = 0.9   # Adjustable
    threshold = np.min(point_density) * rho

    contours = measure.find_contours(
        density.reshape(grid[0].shape),
        level=threshold
    )
