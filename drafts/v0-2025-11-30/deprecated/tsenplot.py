from sklearn.manifold import TSNE
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

embed = KeyedVectors.load_word2vec_format(
    "outputs/track_vectors.txt", binary = False
)

vecs = np.array([embed[k] for k in embed.index_to_key])
labels = embed.index_to_key
X = TSNE(
    n_components = 2, perplexity = 30, random_state = 42
).fit_transform(vecs)

tsne = pd.DataFrame(X, columns = ['x', 'y'])
tsne['node'] = labels
comms = pd.read_csv("outputs/communities.csv")
tsne = tsne.merge(
    comms, left_on = 'node', right_on = 'position', how = 'left'
)

plt.figure(figsize = (8, 6))
plt.scatter(
    tsne['x'], tsne['y'],
    c = tsne['module'],
    cmap = 'tab10', s = 10
)
plt.colorbar(label = "Community")
plt.title("Trajectory embeddings colored by Infomap communities")
plt.show()