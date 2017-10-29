import numpy as np
from sklearn.manifold import TSNE, MDS


avgdist_file = './data/dist_avg.npy'

seed = np.random.seed(0)
avgdist = np.load(avgdist_file)
#avgdist_small = np.copy(avgdist[:100,:100])


def w2c_mds_dec(data, dim=2):
    mds = MDS(n_components=dim, max_iter=3000, eps=1e-9, random_state=seed, dissimilarity='euclidean', n_jobs=-1)
    return mds.fit(data).embedding_


avgdist_vec = np.array(w2c_mds_dec(avgdist, dim=3))

print avgdist_vec
print avgdist_vec.shape
np.save('./avgdist_vec_3.npy', avgdist_vec)
