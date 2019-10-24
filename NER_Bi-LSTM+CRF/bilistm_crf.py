import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

# some helper functions
def argmax(vec):
    # return the argmax as a python int
    # 第1维度上最大值的下标
    # input: tensor([[2,3,4]])
    # output: 2
    _, idx = torch.max(vec,1)
    return idx.item()

def prepare_sequence(seq,to_ix):
    # 文本序列转化为index的序列形式
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def log_sum_exp(vec):
    #compute log sum exp in a numerically stable way for the forward algorithm
    # 用数值稳定的方法计算正演算法的对数和exp
    # input: tensor([[2,3,4]])
    # max_score_broadcast: tensor([[4,4,4]])
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1,-1).expand(1,vec.size()[1])
    return max_score+torch.log(torch.sum(torch.exp(vec-max_score_broadcast)))

START_TAG = "<s>"
END_TAG = "<e>"

# create model
class BiLSTM_CRF(nn.Module):
    def __init__(self,vocab_size, tag2ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF,self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tag2ix = tag2ix
        self.tagset_size = len(tag2ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2, num_layers=1, bidirectional=True)

        # maps output of lstm to tog space
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # matrix of transition parameters
        # entry i, j is the score of transitioning to i from j
        # tag间的转移矩阵，是CRF层的参数
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # these two statements enforce the constraint that we never transfer to the start tag
        # and we never transfer from the stop tag
        self.transitions.data[tag2ix[START_TAG], :] = -10000
        self.transitions.data[:, tag2ix[END_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1,self.hidden_dim//2),
                torch.randn(2, 1,self.hidden_dim//2))

    def _forward_alg(self, feats):
        # to compute partition function
        # 求归一化项的值，应用动态归化算法
        init_alphas = torch.full((1,self.tagset_size), -10000.)# tensor([[-10000.,-10000.,-10000.,-10000.,-10000.]])
        # START_TAG has all of the score
        init_alphas[0][self.tag2ix[START_TAG]] = 0#tensor([[-10000.,-10000.,-10000.,0,-10000.]])

        forward_var = init_alphas

        for feat in feats:
            #feat指Bi-LSTM模型每一步的输出，大小为tagset_size
            alphas_t = []
            for next_tag in range(self.tagset_size):
                # 取其中的某个tag对应的值进行扩张至（1，tagset_size）大小
                # 如tensor([3]) -> tensor([[3,3,3,3,3]])
                emit_score = feat[next_tag].view(1,-1).expand(1,self.tagset_size)
                # 增维操作
                trans_score = self.transitions[next_tag].view(1,-1)
                # 上一步的路径和+转移分数+发射分数
                next_tag_var = forward_var + trans_score + emit_score
                # log_sum_exp求和
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            # 增维
            forward_var = torch.cat(alphas_t).view(1,-1)
        terminal_var = forward_var+self.transitions[self.tag2ix[END_TAG]]
        alpha = log_sum_exp(terminal_var)
        #归一项的值
        return alpha

    def _get_lstm_features(self,sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence),1,-1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self,feats,tags):
        # gives the score of a provides tag sequence
        # 求某一路径的值
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag2ix[START_TAG]], dtype=torch.long), tags])
        for i , feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag2ix[END_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        # 当参数确定的时候，求解最佳路径
        backpointers = []

        init_vars = torch.full((1,self.tagset_size),-10000.)# tensor([[-10000.,-10000.,-10000.,-10000.,-10000.]])
        init_vars[0][self.tag2ix[START_TAG]] = 0#tensor([[-10000.,-10000.,-10000.,0,-10000.]])

        forward_var = init_vars
        for feat in feats:
            bptrs_t = [] # holds the back pointers for this step
            viterbivars_t = [] # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag2ix[END_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag2ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        # 由lstm层计算得的每一时刻属于某一tag的值
        feats = self._get_lstm_features(sentence)
        # 归一项的值
        forward_score = self._forward_alg(feats)
        # 正确路径的值
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score# -(正确路径的分值  -  归一项的值）

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


if __name__ == "__main__":
    EMBEDDING_DIM = 5
    HIDDEN_DIM = 4

    # Make up some training data
    training_data = [(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )]

    word2ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word2ix:
                word2ix[word] = len(word2ix)

    tag2ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, END_TAG: 4}

    model = BiLSTM_CRF(len(word2ix), tag2ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Check predictions before training
    # 输出训练前的预测序列
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], word2ix)
        precheck_tags = torch.tensor([tag2ix[t] for t in training_data[0][1]], dtype=torch.long)
        print(model(precheck_sent))

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word2ix)
            targets = torch.tensor([tag2ix[t] for t in tags], dtype=torch.long)

            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()

    # Check predictions after training
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], word2ix)
        print(model(precheck_sent))

    # 输出结果
    # (tensor(-9996.9365), [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    # (tensor(-9973.2725), [0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 2])