import random
choice = random.choice
import pandas as pd
import re
import jieba
from functools import reduce
from operator import add
from collections import Counter


def create_grammar(grammar_str, split = '=', line_split = '\n'):
    grammar = {}
    for line in grammar_str.split(line_split):
        if not line.strip(): continue
        exp, stmt = line.split(split)
        grammar[exp.strip()] = [ s.split() for s in stmt.split('|') ]
    return grammar

def generate(gram, target):
    if target not in gram: return target   # it means target is terminal expression
    expand = [generate(gram, t) for t in choice(gram[target]) ]
    return ''.join(e if e != '/n' else '\n' for e in expand if e != 'null')

#generate n sentences
def generate_n(n):
    sents =[]
    for i in range(n):
        #print(generate(gram=create_grammar(teacher, split='='), target='order'))
        sen = generate(gram=create_grammar(student, split='='), target='answer')
        sents.append(sen)
    return sents

def token(string):
    return re.findall('\w+', string)

def cut(string): return list(jieba.cut(string))

# compute unigram
def prob_1(word):
    return words_count[word]/len(TOKENS)
#compute bi-gram
def prob_2(word1,word2):
    if word1 + word2 in words_count: return words_count_2[word1+word2]/len(TOKEN_2_GRAM)
    else:
        return (prob_1(word1) + prob_1(word2))/2

# get the probability of generating the sentence
def get_probability(sentence):
    words = cut(sentence)
    sentence_pro = 1
    for i, word in enumerate(words[:-1]):
        next_ = words[i + 1]

        probability = prob_2(word, next_)

        sentence_pro *= probability

    return sentence_pro

# choose the highest probability sentence
def generate_best(n):
    sents = generate_n(n)

    result = []
    for i in range(len(sents)):
        val = get_probability(sents[i])
        print('{}: {}'.format(sents[i], val))
        result.append((val, sents[i]))
    val, sen = sorted(result, key=lambda x: x[0], reverse=True)[0]
    # print(result)
    print("The highest probability sentence is: {}  {} ".format(sen, val))

    return  # return the sentences which has the highest probability

##Part 1: Generate sentences based on the given grammer

# the first grammar
teacher = """
order = 名字 状态 动作 题号 结尾 
名字 = 张三 | 李四 | 王五
状态 = 站着 | 坐着 | 上来
动作 = 解答
题号 = 第 数字 题
数字 = 单个数字 | 数字 单个数字 
单个数字 = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 
结尾 = 。
"""

# the second grammar
student = """
answer = 回答 断句 尊称 断句 题号 答案 结尾
回答 = 到 | 在
断句 = ，
尊称 = 姓 老师
姓 = 于 | 张 | 黄
题号 = 第 数字 题
数字 = 单个数字 | 数字 单个数字 
单个数字 = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 
答案 = 我 结果
结果 = 会 | 不会
结尾 = 。
"""
generate(gram=create_grammar(teacher, split='='), target='order')
generate(gram=create_grammar(student, split='='), target='answer')

# Part 2: train the language model
filename = 'movie_comments.csv'
content = pd.read_csv(filename, encoding='utf-8', low_memory=False)
# content.head()
articles = content['comment'].tolist()
# len(articles)
articles_clean = [''.join(token(str(a)))for a in articles]
articles_words = [
    cut(string) for string in articles_clean[:-1]
]
TOKENS = []
TOKENS = reduce(add, articles_words)
words_count = Counter(TOKENS)
# words_count.most_common(10)
frequience = [f for w,f in words_count.most_common()]
TOKENS = [str(t) for t in TOKENS]
TOKEN_2_GRAM = [''.join(TOKENS[i:i+2]) for i in range(len(TOKENS[:-2]))]
words_count_2 = Counter(TOKENS)


generate_best(10)