import pandas as pd
import numpy as np
import networkx as nx
from infomap import Infomap
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import time
import alphashape
import shapely.geometry as geom
import geopandas as gpd
import multiprocessing as mp

# --- INITIAL STEPS ---
t_start = time.perf_counter()

print(f"[{time.perf_counter()-t_start:.2f}s] Loading data...")
df = pd.read_csv("C:/Users/13647/OneDrive/Desktop/MiMundo/Projects/TrajectoryAnalysis/data/trajectory_adf_zoi.csv")

print(f"[{time.perf_counter()-t_start:.2f}s] Building KDTree...")
coords = df[['lon', 'lat']].values
adf_values = df['ADF'].values
tree = cKDTree(coords)

t0 = time.perf_counter()
print(f"[{time.perf_counter()-t_start:.2f}s] Searching for neighbors...")
sigma = 0.01 
max_dist = sigma * 5
pairs = tree.query_pairs(r=max_dist)

print(f"[{time.perf_counter()-t_start:.2f}s] Building Graph with {len(pairs)} potential edges...")
G = nx.Graph()
G.add_nodes_from(range(len(df)))

# Vectorized edge computation
if pairs:
    pair_array = np.array(list(pairs))
    i_arr, j_arr = pair_array.T
    coords_i = coords[i_arr]
    coords_j = coords[j_arr]
    dists = np.linalg.norm(coords_i - coords_j, axis=1)
    adf_mins = np.minimum(adf_values[i_arr], adf_values[j_arr])
    weights = adf_mins * np.exp(-dists / sigma)
    mask = weights > 0
    edge_list = list(zip(i_arr[mask], j_arr[mask], weights[mask]))
else:
    edge_list = []

G.add_weighted_edges_from(edge_list)

t_graph = time.perf_counter() - t0
print("-" * 60)
print(f"Graph construction complete in {t_graph:.2f}s")

print(f"[{time.perf_counter()-t_start:.2f}s] Running Infomap...")
infomap_wrapper = Infomap("--two-level --silent")
for u, v, data in G.edges(data=True):
    infomap_wrapper.add_link(u, v, data['weight'])
infomap_wrapper.run()

communities = {node.node_id: node.module_id for node in infomap_wrapper.nodes}

# FIX 1: Map the IDs and fill missing ones with -1 (isolated points)
df['community'] = df.index.map(communities).fillna(-1)

# --- 5. LIVE HULL BENCHMARKING ---
print("\n" + "="*50)
print(f"{'ZOI ID':<10} | {'Points':<10} | {'Compute Time':<15}")
print("-" * 50)

# FIX 2: Only look at valid communities (ignore -1 if you don't want hulls for noise)
unique_comms = [c for c in df['community'].unique() if c != -1]

hull_start_time = time.perf_counter()

alpha = 0.01  # tune for scale (smaller alpha = tighter around points)

def compute_hull(comm_id):
    iter_start = time.perf_counter()
    points = df[df['community'] == comm_id][['lon','lat']].values
    num_points = len(points)
    if num_points >= 4:
        poly = alphashape.alphashape(points, alpha)
    else:
        poly = geom.MultiPoint(points)
    iter_end = time.perf_counter()
    print(f"{int(comm_id):<10} | {num_points:<10} | {iter_end - iter_start:>14.6f}s")
    return (comm_id, poly)

# Parallelize hull computation
with mp.Pool(processes=mp.cpu_count()) as pool:
    results = pool.map(compute_hull, unique_comms)

community_polygons = dict(results)

print("-" * 50)
print(f"Total Hull Calculation Time: {time.perf_counter() - hull_start_time:.4f}s")
print("="*50 + "\n")

gdf_list = []
for comm_id, poly in community_polygons.items():
    gdf_list.append(gpd.GeoDataFrame(
        {'community':[comm_id]}, 
        geometry=[poly], 
        crs="EPSG:4326"  # WGS84 lat/lon
    ))

gdf = pd.concat(gdf_list, ignore_index=True)
gdf.to_file("zoi_polygons_low.geojson", driver="GeoJSON")