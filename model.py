import sys
import torch as th
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

class HR_BiLSTM(nn.Module):
    def __init__(self, hidden_size, word_emb, rela_emb):
        super(HR_BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.nb_layers = 1
        # Word Embedding layer
        self.word_embedding = nn.Embedding(word_emb.shape[0], word_emb.shape[1])
        self.word_embedding.weight = nn.Parameter(th.from_numpy(word_emb).float())
        self.word_embedding.weight.requires_grad = False # fix the embedding matrix
        # Rela Embedding layer
        self.rela_embedding = nn.Embedding(rela_emb.shape[0], rela_emb.shape[1])
        self.rela_embedding.weight = nn.Parameter(th.from_numpy(rela_emb).float())
        self.rela_embedding.weight.requires_grad = False # fix the embedding matrix
        # LSTM layer
        self.bilstm_1 = nn.LSTM(word_emb.shape[1], hidden_size, num_layers=self.nb_layers, bidirectional=True)
        self.bilstm_2 = nn.LSTM(hidden_size*2, hidden_size, num_layers=self.nb_layers, bidirectional=True)

        self.cos = nn.CosineSimilarity(1)
        
    def forward(self, question, rela_relation, word_relation):
        question = th.transpose(question, 0, 1)
        rela_relation = th.transpose(rela_relation, 0, 1)
        word_relation = th.transpose(word_relation, 0, 1)

        question = self.word_embedding(question)
        #print('question_emb.shape', question.shape)
        rela_relation = self.rela_embedding(rela_relation)
        #print('rela_relation_emb.shape', rela_relation.shape)
        word_relation = self.word_embedding(word_relation)
        #print('word_relation_emb.shape', word_relation.shape)
        #print()

#        self.bilstm_1.flatten_parameters()
        question_out_1, question_hidden = self.bilstm_1(question)
        #print('question_out_1.shape', question_out_1.shape)
#        self.bilstm_2.flatten_parameters()
        question_out_2, _ = self.bilstm_2(question_out_1)
        #print('question_out_2.shape', question_out_2.shape)
        
        # 1st way of Hierarchical Residual Matching
        q12 = question_out_1 + question_out_2
        q12 = q12.permute(1, 2, 0)
        #print('q12.shape', q12.shape)
        question_representation = nn.MaxPool1d(q12.shape[2])(q12) 
        question_representation = question_representation.squeeze(2)
        # 2nd way of Hierarchical Residual Matching
        #q1_max = nn.MaxPool1d(question_out_1.shape[2])(question_out_1)
        #q2_max = nn.MaxPool1d(question_out_2.shape[2])(question_out_2)
        #question_representation = q1_max + q2_max
        #print('question_representation.shape', question_representation.shape) 

        #print()
#        self.bilstm_1.flatten_parameters()
        word_relation_out, word_relation_hidden = self.bilstm_1(word_relation)
        #print('word_relation_out.shape', word_relation_out.shape)
#        self.bilstm_1.flatten_parameters()
        rela_relation_out, rela_relation_hidden = self.bilstm_1(rela_relation, word_relation_hidden)
        #print('rela_relation_out.shape', rela_relation_out.shape)
        r = th.cat([rela_relation_out, word_relation_out], 0)
        r = r.permute(1, 2, 0)
        #print('r.shape', r.shape)
        relation_representation = nn.MaxPool1d(r.shape[2])(r)
        relation_representation = relation_representation.squeeze(2)
        #print('relation_representation.shape', relation_representation.shape)

        score = self.cos(question_representation, relation_representation)
        #print('score.shape', score.shape)
        return score
    
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.ques_embedding = nn.Embedding(args.ques_embedding.shape[0], args.ques_embedding.shape[1])
        self.ques_embedding.weight.requires_grad = False
        self.ques_embedding.weight = nn.Parameter(th.from_numpy(args.ques_embedding).float())
        self.rela_text_embedding = nn.Embedding(args.rela_text_embedding.shape[0], args.rela_text_embedding.shape[1])
        self.rela_text_embedding.weight.requires_grad = False
        self.rela_text_embedding.weight = nn.Parameter(th.from_numpy(args.rela_text_embedding).float())
        self.rela_embedding = nn.Embedding(args.rela_vocab_size, args.rela_text_embedding.shape[1])
        self.rnn = nn.LSTM(input_size=args.emb_size, hidden_size=args.hidden_size,
                          num_layers=args.num_layers, batch_first=False,
                          dropout=args.dropout_rate, bidirectional=True)
        self.rnn2 = nn.LSTM(input_size=args.hidden_size*2, hidden_size=args.hidden_size,
                          num_layers=args.num_layers, batch_first=False,
                          dropout=args.dropout_rate, bidirectional=True)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.args = args
        self.cos = nn.CosineSimilarity(dim=1)
        self.tanh = nn.Tanh()
        return

    def forward(self, ques_x, rela_text_x, rela_x):
        ques_x = th.transpose(ques_x, 0, 1)
        rela_text_x = th.transpose(rela_text_x, 0, 1)
        rela_x = th.transpose(rela_x, 0, 1)

        ques_x = self.ques_embedding(ques_x)
        rela_text_x = self.rela_text_embedding(rela_text_x)
        rela_x = self.rela_embedding(rela_x)

        ques_hs1, hidden_state = self.encode(ques_x)
        rela_hs, hidden_state = self.encode(rela_x, hidden_state)
        rela_text_hs, hidden_state = self.encode(rela_text_x, hidden_state)

        h_0 = Variable(th.zeros([self.args.num_layers*2, len(ques_x[0]), self.args.hidden_size])).cuda()
        c_0 = Variable(th.zeros([self.args.num_layers*2, len(ques_x[0]), self.args.hidden_size])).cuda()
        ques_hs2, _ = self.rnn2(ques_hs1, (h_0, c_0)) 

        ques_hs = ques_hs1 + ques_hs2
        ques_hs = ques_hs.permute(1, 2, 0)
        ques_h = F.max_pool1d(ques_hs, kernel_size=ques_hs.shape[2], stride=None)
        rela_hs = th.cat([rela_hs, rela_text_hs], 0)
        rela_hs = rela_hs.permute(1, 2, 0)
        rela_h = F.max_pool1d(rela_hs, kernel_size=rela_hs.shape[2], stride=None)

        ques_h = ques_h.squeeze(2)
        rela_h = rela_h.squeeze(2)

        output = self.cos(ques_h, rela_h)
        return output

    def encode(self, input, hidden_state=None, return_sequence=True):
        if hidden_state==None:
            h_0 = Variable(th.zeros([self.args.num_layers*2, len(input[0]), self.args.hidden_size])).cuda()
            c_0 = Variable(th.zeros([self.args.num_layers*2, len(input[0]), self.args.hidden_size])).cuda()
        else:
            h_0, c_0 = hidden_state
        h_input = h_0
        c_input = c_0
        outputs, (h_output, c_output) = self.rnn(input, (h_0, c_0))
        if return_sequence == False:
            return outputs[-1], (h_output, c_output)
        else:
            return outputs, (h_output, c_output)

class ABWIM(nn.Module):
    def __init__(self, dropout_rate, hidden_size, word_emb, rela_emb, q_len, r_len):
        super(ABWIM, self).__init__()
        self.hidden_size = hidden_size
        self.nb_layers = 1
        self.nb_filters = 100
        self.dropout = nn.Dropout(dropout_rate)
        # Word Embedding layer
        self.word_embedding = nn.Embedding(word_emb.shape[0], word_emb.shape[1])
        self.word_embedding.weight = nn.Parameter(th.from_numpy(word_emb).float())
        self.word_embedding.weight.requires_grad = False # fix the embedding matrix
        # Rela Embedding layer
        self.rela_embedding = nn.Embedding(rela_emb.shape[0], rela_emb.shape[1])
        self.rela_embedding.weight = nn.Parameter(th.from_numpy(rela_emb).float())
        self.rela_embedding.weight.requires_grad = False # fix the embedding matrix
        # LSTM layer
        self.bilstm = nn.LSTM(word_emb.shape[1], 
                              hidden_size, 
                              num_layers=self.nb_layers, 
                              bidirectional=True, 
                              dropout=dropout_rate)
        # Attention
        self.W = Variable(th.rand((hidden_size*2, hidden_size*2))).cuda()
        # CNN layer
        self.cnn_1 = nn.Conv1d(hidden_size*4, self.nb_filters, 1)
        self.cnn_2 = nn.Conv1d(hidden_size*4, self.nb_filters, 3)
        self.cnn_3 = nn.Conv1d(hidden_size*4, self.nb_filters, 5)
        self.activation = nn.ReLU()
        self.maxpool_1 = nn.MaxPool1d(q_len)
        self.maxpool_2 = nn.MaxPool1d(q_len-2)
        self.maxpool_3 = nn.MaxPool1d(q_len-4)
        self.linear = nn.Linear(self.nb_filters, 1, bias=False)

    def init_hidden(self, batch_size):
        return (Variable(th.rand(2, batch_size, self.hidden_size)).cuda(),
                Variable(th.rand(2, batch_size, self.hidden_size)).cuda())
        
    def forward(self, question, rela_relation, word_relation):
        question = th.transpose(question, 0, 1)
        rela_relation = th.transpose(rela_relation, 0, 1)
        word_relation = th.transpose(word_relation, 0, 1)

        question = self.word_embedding(question)
        question = self.dropout(question)
        #print('question_emb.shape', question.shape) # [13, 5861, 300]
        rela_relation = self.rela_embedding(rela_relation)
        rela_relation = self.dropout(rela_relation)
        #print('rela_relation_emb.shape', rela_relation.shape) # [12, 5861, 300]
        word_relation = self.word_embedding(word_relation)
        word_relation = self.dropout(word_relation)
        #print('word_relation_emb.shape', word_relation.shape) # [21, 5861, 300]

#        self.bilstm.flatten_parameters()
        question_out, _ = self.bilstm(question, self.init_hidden(question.shape[1]))
        #print('question_out.shape', question_out.shape) # [13, 5861, 200]
        question_out = question_out.permute(1,2,0)
        #print('question_out.shape', question_out.shape) # [5861, 200, 13]
        word_relation_out, word_relation_hidden = self.bilstm(word_relation, self.init_hidden(word_relation.shape[1]))
        rela_relation_out, _ = self.bilstm(rela_relation, word_relation_hidden)
        relation = th.cat([rela_relation_out, word_relation_out], 0)
        #print('relation.shape', relation.shape) # [33, 5861, 200]
        relation = relation.permute(1,0,2)
        #print('relation.shape', relation.shape) # [5861, 33, 200]

        # attention layer
        energy = th.matmul(relation, self.W)
        #print('energy.shape', energy.shape) # [12599, 33, 200]
        energy = th.matmul(energy, question_out)
        #print('energy.shape', energy.shape) # [12599, 33, 13]
        alpha = F.softmax(energy, dim=-1)
        #print('alpha.shape', alpha.shape) # [12599, 33, 13]
        alpha = alpha.unsqueeze(3)
        #print('alpha.shape', alpha.shape) # [12599, 33, 13, 1]
        relation = relation.unsqueeze(2)
        #print('relation.shape', relation.shape) # [12599, 33, 1, 200]
        atten_relation = alpha * relation
        #print('atten_relation.shape', atten_relation.shape) # [12599, 33, 13, 200]
        atten_relation = th.sum(atten_relation, 1)
        #print('atten_relation.shape', atten_relation.shape) # [12599, 13, 200]
        atten_relation = atten_relation.permute(0, 2, 1)
        #print('atten_relation.shape', atten_relation.shape) # [12599, 200, 13]
        M = th.cat((question_out, atten_relation), 1)
        #print('M.shape', M.shape) # [12599, 400, 13]
        h1 = self.maxpool_1(self.activation(self.cnn_1(M)))
        h1 = self.dropout(h1)
        h2 = self.maxpool_2(self.activation(self.cnn_2(M)))
        h2 = self.dropout(h2)
        h3 = self.maxpool_3(self.activation(self.cnn_3(M)))
        h3 = self.dropout(h3)
        h = th.cat((h1, h2, h3),2)
        h = th.max(h, 2)[0]
        score = self.linear(h).squeeze()
        return score
    
