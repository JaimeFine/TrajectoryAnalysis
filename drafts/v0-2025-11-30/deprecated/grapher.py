import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import json

wv = KeyedVectors.load_word2vec_format(
    "outputs/track_vectors.txt", binary=False
)

with open("data/flight_routes.geojson") as flights:
    track = json.load(flights)

edges = []

for feature in track["features"]:
    coords = feature["geometry"]["coordinates"]
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        x = f"{lon1:.3f}_{lat1:.3f}"
        y = f"{lon2:.3f}_{lat2:.3f}"
        edges.append((x, y))

df = pd.DataFrame(
    edges, columns = ["dot_origin", "dot_destination"]
)
df = df.groupby(
    ["dot_origin", "dot_destination"]
).size().reset_index(name = "freq")

def sim(x, y):
    if x in wv and y in wv:
        return float(cosine_similarity(wv[x].reshape(1,-1), wv[y].reshape(1,-1))[0,0])
    else:
        return 0.0

df['sim'] = df.apply(
    lambda r: sim(r['dot_origin'], r['dot_destination']),
    axis=1
)
df['sim_norm'] = (df['sim'] + 1) / 2
df['weight'] = df['freq'] * df['sim_norm']

G = nx.DiGraph()
for idx, row in df.iterrows():
    G.add_edge(
        row['dot_origin'],
        row['dot_destination'],
        weight=row['weight'],
        freq=row['freq']
    )

df.to_csv("outputs/graph_edges.csv", index=False)
nx.write_gexf(G, "outputs/track_graph.gexf")
