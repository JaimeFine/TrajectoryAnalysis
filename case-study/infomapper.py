import pandas as pd
import numpy as np
import networkx as nx
from infomap import Infomap
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/13647/OneDrive/Desktop/MiMundo/Projects/TrajectoryAnalysis/data/trajectory_adf_zoi.csv")

G = nx.Graph()

for idx, row in df.iterrows():
    G.add_node(idx, lon=row['lon'], lat=row['lat'], adf=row['ADF'])

sigma = 0.01    # A spatial decay parameter
for i, row_i in df.iterrows():
    for j, row_j in df.iterrows():
        if i >= j:
            continue
        # Edge weight = min(ADF_i, ADF_j) * spatial decay
        distance = np.sqrt(
            (row_i['lon'] - row_j['lon'])**2 + (row_i['lat'] - row_j['lat'])**2
        )
        weight = min(row_i['ADF'], row_j['ADF']) * np.exp(-distance / sigma)
        if weight > 0:
            G.add_edge(i, j, weight=weight)

infomap_wrapper = Infomap("--two-level --silent")

for u, v, data in G.edges(data=True):
    infomap_wrapper.add_link(u, v, data['weight'])

infomap_wrapper.run()

communities = {}
for node in infomap_wrapper.nodes:
    communities[node.node_id] = node.module_id

df['community'] = df.index.map(communities)

community_hulls = {}
for comm_id in df['community'].unique():
    points = df[df['community'] == comm_id][['lon', 'lat']].values
    if len(points) >= 3:
        hull = ConvexHull(points)
        community_hulls[comm_id] = points[hull.vertices]
    else:
        community_hulls[comm_id] = points

plt.figure(figsize=(10,8))
for comm_id, hull_points in community_hulls.items():
    plt.fill(hull_points[:,0], hull_points[:,1], alpha=0.3, label=f'ZOI {comm_id}')
plt.scatter(df['lon'], df['lat'], c='k', s=10)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Trajectory-conditioned ZOIs via Infomap")
plt.legend()
plt.show()