# 初始化词典
tag2id, id2tag = {},{}
word2id, id2word = {},{}

#建立词典
for line in open("traindata.txt"):
    items = line.split("/")
    word, tag = items[0],items[1].rstrip()

    if word not in word2id:
        word2id[word]=len(word2id)
        id2word[len(id2word)] = word
    if tag not in tag2id:
        tag2id[tag] = len(tag2id)
        id2tag[len(id2tag)] = tag


#print(word2id)
#初始化参数
M = len(word2id)
N = len(tag2id)

import numpy as np
pi = np.zeros(N)
B = np.zeros((N,M))#发射
A = np.zeros((N,N))#状态转移

#统计参数
prev_tag = ""
for line in open("traindata.txt"):
    items = line.split("/")
    wordId, tagId = word2id[items[0]],tag2id[items[1].rstrip()]
    if prev_tag == "":
        pi[tagId] += 1
        B[tagId][wordId] += 1
    else:
        B[tagId][wordId] += 1
        A[tag2id[prev_tag]][tagId] += 1

    if items[0] == ".":
        prev_tag=""
    else:
        prev_tag = items[1].rstrip()

#转化成概率的形式
pi = pi/sum(pi)
for i in range(N):
    A[i] /= sum(A[i])
    B[i] /= sum(B[i])

#print(f"pi:{pi}")

def log(v):
    if v == 0:
        return np.log(v+0.000001)
    return np.log(v)

#维特比算法
def viterbi(x,pi,A, B):
    x = [word2id[word] for word in x.split(" ")]
    T = len(x)

    dp = np.zeros((T,N))
    ptr = np.zeros((T,N),dtype=int)

    for j in range(N):
        dp[0][j] = log(pi[j]) + log(B[j][x[0]])

    for i in range(1,T):
        for j in range(N):
            dp[i][j] = -9999999
            for k in range(N):
                score = dp[i-1][k]+log(A[k][j])+log(B[j][x[i]])
                if score > dp[i][j]:
                    dp[i][j] = score
                    ptr[i][j] = k

    best_seq = [0]*T
    best_seq[T-1] = np.argmax(dp[T-1])

    for i in range(T-2,-1,-1):
        best_seq[i] = ptr[i+1][best_seq[i+1]]


    for i in range(len(best_seq)):
        print(id2tag[best_seq[i]])

if __name__ == "__main__":
    x = "keep new to everything"
    viterbi(x, pi, A, B)