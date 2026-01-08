import json
import os
import numpy as np

def load_data(path):
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_cosine_similarities_mean(vec, vecs):
    vec = np.array(vec)
    vecs = np.array(vecs)
    dot_products = vecs.dot(vec)
    norm_vec = np.linalg.norm(vec)
    norms_vecs = np.linalg.norm(vecs, axis=1)
    cosine_similarities = dot_products / (norm_vec * norms_vecs)
    return np.mean(cosine_similarities)