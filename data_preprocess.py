''' Training Data:
    KBQA_RE_data/webqsp_relations/relations.txt
    KBQA_RE_data/webqsp_relations/WebQSP.RE.train.with_boundary.withpool.dlnlp.txt
    KBQA_RE_data/webqsp_relations/WebQSP.RE.test.with_boundary.withpool.dlnlp.txt
'''
import sys
import os
import time
import gensim
import numpy as np

import torch
from torch.autograd import Variable
import math
class DataManager:
    def __init__(self):
        self.train_data_path = 'KBQA_RE_data/webqsp_relations/WebQSP.RE.train.with_boundary.withpool.dlnlp.txt'
        self.test_data_path = 'KBQA_RE_data/webqsp_relations/WebQSP.RE.test.with_boundary.withpool.dlnlp.txt'
        self.word_embedding_path = 'KBQA_RE_word_emb_300d.txt'
        self.rela_embedding_path = 'KBQA_RE_rela_emb_300d.txt'
        #self.word_embedding_path = 'KBQA_RE_word_emb_300d_last.txt'
        #self.rela_embedding_path = 'KBQA_RE_rela_emb_300d_last.txt'
        self.emb_dim = 300
        self.relations_map = {}
        self.word_dic = {}
        self.word_embedding = []
        self.rela_dic = {}
        self.rela_embedding = []

        self.relations_map = self.load_relations_map('KBQA_RE_data/webqsp_relations/relations.txt')
        print('Original training questions: 3116')
        print('Original testing questions: 1649')
        train_data = self.gen_train_data(self.train_data_path) # len: 3116 (questions)
        test_data = self.gen_train_data(self.test_data_path) # len: 1649 (questions)
        print()

        if not os.path.isfile(self.word_embedding_path):
            print(self.word_embedding_path, 'not exist!')
            self.save_embeddings(train_data+test_data)
            print()
        self.word_dic, self.word_embedding = self.load_embeddings(self.word_embedding_path)
        self.rela_dic, self.rela_embedding = self.load_embeddings(self.rela_embedding_path)
        print()        
        token_train_data = self.tokenize_train_data(train_data)
        token_test_data = self.tokenize_train_data(test_data)
        print()
        self.maxlen_q, maxlen_pos_r, maxlen_pos_w, maxlen_neg_r, maxlen_neg_w = self.find_maxlength(token_train_data)
        self.maxlen_r = max(maxlen_pos_r, maxlen_neg_r)
        self.maxlen_w = max(maxlen_pos_w, maxlen_neg_w)
        #print('self.maxlen_q, self.maxlen_pos_r, self.maxlen_pos_w, self.maxlen_neg_r, self.maxlen_neg_w')
        #print(self.maxlen_q, self.maxlen_pos_r, self.maxlen_pos_w, self.maxlen_neg_r, self.maxlen_neg_w)
        self.token_train_data = self.pad_train_data(token_train_data)
        self.token_test_data = self.pad_train_data(token_test_data)
        print('Check token result')
        print(len(train_data), len(self.token_train_data))
        print(len(train_data[0]), len(self.token_train_data[0]))
        print('question, pos_relations, pos_words, neg_relations, neg_words')
        print(train_data[0][0])
        print(self.token_train_data[0][0])
        print()
    
    def idx2word(self, id_sentence, id_type='word'):
        if id_type == 'relation':
            dic = self.rela_dic
        else:
            dic = self.word_dic
        word_sentence = []
        for idx in id_sentence:
            if idx == 0:
                continue
            word_sentence.append(dic[idx])
        return ' '.join(word_sentence)

    def gen_train_data(self, path):
        ''' Return training_data_list [[(question, pos_relas, pos_words, neg_relas, neg_words) * neg_size] * q_size]
            Note that neg may be empty!
        '''
        data_list = []
        #pos_gt_1_counter = 0
        print('Load', path)
        #start = time.time()
        idx = 0
        with open(path) as infile:
            for line in infile: 
                q_list = []
                tokens = line.strip().split('\t')
                pos_relations = tokens[0]
                neg_relations = tokens[1]
                question = tokens[2].replace('$ARG1','').replace('$ARG2','').strip().split(' ')
                #if len(pos_relations.split(' ')) != 1:
                #    pos_gt_1_counter += 1
                for pos in pos_relations.split(' '):
                    pos_relas, pos_words = self.split_relation(pos)
                    for neg in neg_relations.split(' '):
                        # skip blank neg_relation
                        if neg == '1797':
                            continue
                        neg_relas, neg_words = self.split_relation(neg)
                        q_list.append((question, pos_relas, pos_words, neg_relas, neg_words))
                        #print(q_list)
                        #sys.exit()
                if len(q_list) > 0:
                    data_list.append(q_list)
        print('Filter out questions without negative training samples.')
        print(f'Train data length:{len(data_list)}')
        #print(f'Time elapsed:{time.time()-start:.2f}')
        #print('pos_gt_1_counter', pos_gt_1_counter)
        return data_list

    def split_relation(self, relation_id):
        '''Split relations (only take the last part as relation name); Split relation names to words;
        '''
        rela_list = []
        word_list = []
        relation_names = self.relations_map[int(relation_id)]
        for relation_name in relation_names.split('..'):
            #last_name = relation_name.split('.')[-1]
            #rela_list.append(last_name)
            #for word in last_name.split('_'):
            #    word_list.append(word)
            for relation_split in relation_name.split('.'):
                rela_list.append(relation_split)
                for word in relation_split.split('_'):
                    word_list.append(word)
        return rela_list, word_list

    def find_unique(self, data):
        words = set()
        relas = set()
        #start = time.time()
        for idx, q_data in enumerate(data, 1):
            print('\r# of questions', idx, end='')
            try:
                words |= set(q_data[0][0])
                for data_obj in q_data:
                    relas |= set(data_obj[1]) | set(data_obj[3])
                    words |= set(data_obj[2]) | set(data_obj[4])
            except:
                print(idx, q_data)
        print()
        #relas.remove('')
        if '' in words:
            words.remove('')
        print(f'There are {len(relas)} unique relations and {len(words)} unique words.')
        #print(f'Time elapsed:{time.time()-start:.2f}')
        return relas, words

    def load_word_embedding_from_gensim(self, input_path):
        print('Load pretrain word embedding from', input_path)
        #start = time.time()
        model = gensim.models.Word2Vec.load_word2vec_format(input_path, binary=True)
        #print(f'Time elapsed:{time.time()-start:.2f}') 
        return model

    def load_embeddings(self, path):
        vocab_dic = {}
        embedding = []
        print('Load embedding from', path)
        with open(path) as infile:
            for line in infile:
                tokens = line.strip().split()
                vocab_dic[tokens[0]] = len(vocab_dic)
                embedding.append([float(x) for x in tokens[1:]]) 
        embedding = np.array(embedding)
        print('vocab size', len(vocab_dic))
        print('emb shape', embedding.shape)
        if embedding.shape[1] != self.emb_dim:
            print('Load embedding error!')
            sys.exit()
        return vocab_dic, embedding

    def save_embeddings(self, data):
        rela_set, word_set = self.find_unique(data)

        # Load 300 dim pretrained word2vec embeddings trained on GoogleNews. 
        # To be more efficient, only load words contains in training/testing data.
        exception_counter = 0
        input_w2v_path = '/corpus/wordvector/word2vec/GoogleNews-vectors-negative300.bin'
        Word2Vec_embedding = self.load_word_embedding_from_gensim(input_w2v_path)
        word_list = ['PADDING','<e>','<unk>']
        embedding_dic = {}
        embedding_dic['PADDING'] = np.array([0.0] * self.emb_dim)
        embedding_dic['<e>'] = np.random.uniform(low=-0.25, high=0.25, size=(self.emb_dim,))
        embedding_dic['<unk>'] = np.random.uniform(low=-0.25, high=0.25, size=(self.emb_dim,))
        print('Dump word embedding to', self.word_embedding_path)
        with open(self.word_embedding_path, 'w') as outfile:
            for word in word_list:
                outfile.write(word+' ')
                outfile.write(' '.join(str(v) for v in embedding_dic[word]))
                outfile.write('\n')
            for word in list(word_set):
                if word in Word2Vec_embedding:
                    outfile.write(word+' ')
                    outfile.write(' '.join(str(v) for v in Word2Vec_embedding[word]))
                    outfile.write('\n')
                else:
                    exception_counter += 1
        print(f'{exception_counter} words not found.') #207

        # store relation_dic, relation_embedding
        exception_counter = 0
        rela_list = list(rela_set)
        rela_embedding = np.random.uniform(low=-0.25, high=0.25, size=(len(rela_set), self.emb_dim,))
        print('Dump relation embedding to', self.rela_embedding_path)
        with open(self.rela_embedding_path, 'w') as outfile:
            for idx, relation in enumerate(rela_list):
                try:
                    outfile.write(relation+' ')
                    outfile.write(' '.join(str(v) for v in rela_embedding[idx]))
                    outfile.write('\n')
                except:
                    exception_counter += 1
                    print(f'"{relation}" got exception!')
        print(f'{exception_counter} relations not found.') 
        return 0

    def tokenize_train_data(self, data):
        token_data = []
        #start = time.time()
        for idx, q_data in enumerate(data):
            token_q_data = []
            question = list(map(lambda x: self.word_dic[x] if x in self.word_dic else self.word_dic['<unk>'], q_data[0][0]))
            #question = list(map(lambda x: self.tokenize(x, self.word_dic), q_data[0][0]))
            for data_obj in q_data:
                pos_relas = list(map(lambda x: self.rela_dic[x], data_obj[1]))
                pos_words = list(map(lambda x: self.word_dic[x] if x in self.word_dic else self.word_dic['<unk>'], data_obj[2]))
                neg_relas = list(map(lambda x: self.rela_dic[x], data_obj[3]))
                neg_words = list(map(lambda x: self.word_dic[x] if x in self.word_dic else self.word_dic['<unk>'], data_obj[4]))
                token_q_data.append((question, pos_relas, pos_words, neg_relas, neg_words))
            token_data.append(token_q_data)
        print(f'Token data length:{len(token_data)}')
        #print(f'Time elapsed:{time.time()-start:.2f}')
        return token_data

    def tokenize(self, word, dic):
        if word in dic:
            return dic[word]
        else:
            return dic['<unk>']
    
    def pad_train_data(self, data):
        ''' Input: training_data_list [[(question, pos_relas, pos_words, neg_relas, neg_words) * neg_size] * q_size]
        '''
        padded_data = []
        for q_data in data:
            padded_q_data = []
            for obj in q_data:
                q = self.pad_obj(self.maxlen_q, obj[0])
                p_rela = self.pad_obj(self.maxlen_r, obj[1])
                p_word = self.pad_obj(self.maxlen_w, obj[2])
                n_rela = self.pad_obj(self.maxlen_r, obj[3])
                n_word = self.pad_obj(self.maxlen_w, obj[4])
                padded_q_data.append((
                    q, p_rela, p_word, n_rela, n_word
                ))
            padded_data.append(padded_q_data)
        return padded_data

    def pad_obj(self, max_len, sentence):
        if max_len >= len(sentence):
            return [0]*(max_len-len(sentence)) + sentence
        else:
            return sentence[:max_len]

    def find_maxlength(self, data):
        maxlen_q, maxlen_pos_r, maxlen_pos_w, maxlen_neg_r, maxlen_neg_w = 0, 0, 0, 0, 0
        for q_data in data:
            for obj in q_data:
                if len(obj[0]) > maxlen_q:
                    maxlen_q = len(obj[0])
                if len(obj[1]) > maxlen_pos_r:
                    maxlen_pos_r = len(obj[1])
                if len(obj[2]) > maxlen_pos_w:
                    maxlen_pos_w = len(obj[2])
                if len(obj[3]) > maxlen_neg_r:
                    maxlen_neg_r = len(obj[3])
                if len(obj[4]) > maxlen_neg_w:
                    maxlen_neg_w = len(obj[4])
        #print(maxlen_q, maxlen_pos_r, maxlen_pos_w, maxlen_neg_r, maxlen_neg_w)
        return maxlen_q, maxlen_pos_r, maxlen_pos_w, maxlen_neg_r, maxlen_neg_w
    
    def load_relations_map(self, path):
        ''' Return self.relations_map = {idx:relation_names}
        '''
        relations_map = {}
        with open(path) as infile:
            idx = 1
            for line in infile:
                relations_map[idx] = line.strip()
                idx += 1
        return relations_map
    
    def question_preprocess(self, path):
        '''Return normalized_question_list from WebQSP
        '''
        q_list = []
        print('Load', path)
        start = time.time()
        WebQSP = json.load(open(path))
        questions = WebQSP['Questions']
        q_list.extend(data['ProcessedQuestion'] for data in questions)
        print(f'Length:{len(q_list)}')
        print(f'Time elapsed:{time.time()-start:.2f}')
        return q_list

def batchify(data, batch_size):
    ''' Input: training_data_list [[(question, pos_relas, pos_words, neg_relas, neg_words) * neg_size] * q_size]
        Return: [[(question, pos_relas, pos_words, neg_relas, neg_words)*neg_size] * batch_size] * nb_batch]
    '''
    nb_batch = math.ceil(len(data) / batch_size)
    batch_data = [data[idx*batch_size:(idx+1)*batch_size] for idx in range(nb_batch)]
    print('nb_batch', len(batch_data), 'batch_size', len(batch_data[0]))
    return batch_data


if __name__ == '__main__': 
    data = DataManager()
