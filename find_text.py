from bs4 import BeautifulSoup, Tag
from soupselect import select
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

import requests
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def find_similar(tfidf_matrix, index, top_n = 5):
    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    return [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]

corpus = []
file_mapping = {}
for idx, file in enumerate(glob.glob("data/*")):
    page = BeautifulSoup(open(file, "r"), "html.parser")
    content = select(page, "div.post-body")[0].text
    corpus.append(content)
    file_mapping[idx] = file

tf = TfidfVectorizer(analyzer='word', min_df = 0, stop_words = 'english')
tfidf_matrix =  tf.fit_transform(corpus)

reducer = TruncatedSVD(n_components=100)
reducer.fit(tfidf_matrix)
svd_all = reducer.transform(tfidf_matrix)

model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
new_all =  model.fit_transform(svd_all)

df = pd.DataFrame(None)
df["page"] =  [file_mapping[idx] for idx, value in enumerate(new_all)]
df["X coordinate"] = [x[0] for x in new_all]
df["Y coordinate"] = [x[1] for x in new_all]

def onpick3(event):
    ind = event.ind
    print "ind:{0}, x:{1}, y:{2}, id: {3}".format(ind, df["X coordinate"][ind], df["Y coordinate"][ind], df["page"][ind])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(df["X coordinate"], df["Y coordinate"], picker = True)
fig.canvas.mpl_connect('pick_event', onpick3)

plt.show()
