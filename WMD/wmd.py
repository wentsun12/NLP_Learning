import numpy as np
from nltk.corpus import stopwords
from nltk import download
from collections import defaultdict
from itertools import product
import pulp
from scipy.spatial.distance import euclidean

download('stopwords')  # Download stopwords list.
stop_words = stopwords.words('english')

# make embeddings_index
def embed_matrix():
    glovefile = open("glove.6B.100d.txt", "r", encoding="utf-8")
    lines = glovefile.readlines()
    embeddings_index = {}
    for line in lines:
        values = line.split()
        word = values[0]
        embed = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embed

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def tokenize(sent):
    # split words
    token = sent.lower().split()
    # remove stop words
    token = [w for w in token if w not in stop_words]
    return token

def nBOW(tokens):
    # initialization
    cntdict = defaultdict(lambda: 0)
    for token in tokens:
        cntdict[token] += 1
    totalcnt = sum(cntdict.values())
    return {token: float(cnt) / totalcnt for token, cnt in cntdict.items()}


def WMD(sent1, sent2):
    # compute the word distance
    embedding_index = embed_matrix()
    # split sentences and romove stop words
    tokens1 = tokenize(sent1)
    tokens2 = tokenize(sent2)
    # list all words
    all_tokens = list(set(tokens1 + tokens2))
    # compute nbow values
    first_sent_buckets = nBOW(tokens1)
    second_sent_buckets = nBOW(tokens2)
    # set T_matrix
    T = pulp.LpVariable.dicts('T_matrix', list(product(all_tokens, all_tokens)), lowBound=0)
    #
    wmd_dist = pulp.LpProblem('WMD', sense=pulp.LpMinimize)
    wmd_dist += pulp.lpSum(
        T[token1, token2] * euclidean(embedding_index[token1], embedding_index[token2]) for token1, token2 in
        product(all_tokens, all_tokens))

    for token2 in second_sent_buckets:
        wmd_dist += pulp.lpSum([T[token1, token2] for token1 in first_sent_buckets]) == second_sent_buckets[token2]
    for token1 in first_sent_buckets:
        wmd_dist += pulp.lpSum([T[token1, token2] for token2 in second_sent_buckets]) == first_sent_buckets[token1]

    wmd_dist.solve()
    # print(pulp.value(wmd_dist.objective))

    return wmd_dist

sent1 = ['people like this car',
         'i have to go to supermarket today',
         'there are books on the table',
         'he likes reading']
sent2 = ['those guys enjoy driving that',
         'we need to buy something',
         'there are boxes under the table',
         'there is no book']
for s1,s2 in zip(sent1, sent2):
    dist = WMD(s1, s2)
    print(pulp.value(dist.objective))
    for v in dist.variables():
        if v.varValue != 0:
            print(v.name, '=', v.varValue)

