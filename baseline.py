import sys
import time
import argparse
import math
import numpy as np
from random import shuffle
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
from torch.autograd import Variable

from data_preprocess import DataManager
from model import HR_BiLSTM
from model import ABWIM

parser = argparse.ArgumentParser()
# setting
parser.add_argument('--learning_rate', type=float, default=0.1) # [0.1/0.5/1.0/2.0]
parser.add_argument('--hidden_size', type=int, default=50) # [50/100/200/400]
parser.add_argument('--optimizer', type=str, default='Adadelta')
parser.add_argument('--epoch_num', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=64)
#parser.add_argument('--margin', type=float, default=0.1)
#parser.add_argument('--dropout', type=float, default=0.35)
parser.add_argument('--earlystop_tolerance', type=int, default=5)
parser.add_argument('--save_model_path', type=str, default='')
parser.add_argument('--pretrain_model', type=str, default=None)
args = parser.parse_args()
args.save_model_path = ''

torch.cuda.manual_seed(1234)
#####################################################################
# Load data
#####################################################################
def batchify(data, batch_size):
    ''' Input: training_data_list [[(question, pos_relas, pos_words, neg_relas, neg_words) * neg_size] * q_size]
        Return: [[(question, pos_relas, pos_words, neg_relas, neg_words)*neg_size] * batch_size] * nb_batch]
    '''
    nb_batch = math.ceil(len(data) / batch_size)
    batch_data = [data[idx*batch_size:(idx+1)*batch_size] for idx in range(nb_batch)]
    print('nb_batch', len(batch_data), 'batch_size', len(batch_data[0]))
    return batch_data

corpus = DataManager()
# shuffle training data
#shuffle(corpus.token_train_data)

## split training data to train and validation
#split_num = int(0.9*len(corpus.token_train_data))
#print('split_num=', split_num)
#train_data = corpus.token_train_data[:split_num]
#val_data = corpus.token_train_data[split_num:]
#print('training data length:', len(train_data))
#print('validation data length:', len(val_data))
#print('test data length:', len(corpus.token_test_data))
#print()
#
#''' batchify questions
#    uncomment Line 119, 120
#'''
##batch_train_data = batchify(train_data, args.batch_size)
#''' batchify train_objs
#    uncomment Line 121
#'''
#flat_train_data = [obj for q_obj in train_data for obj in q_obj]
#print('len(flat_train_data)', len(flat_train_data))
#shuffle(flat_train_data)
#batch_train_data = batchify(flat_train_data, args.batch_size)
#
#val_data = batchify(val_data, 64)
test_data = batchify(corpus.token_test_data, 64)
print()
#####################################################################
# Build model
#####################################################################
#loss_function = nn.MarginRankingLoss(margin=args.margin)
loss_function = nn.MarginRankingLoss()
print('Build model')
#q_len = corpus.maxlen_q
#r_len = corpus.maxlen_w + corpus.maxlen_r
#print('q_len', q_len, 'r_len', r_len)
#model = ABWIM(args.dropout, args.hidden_size, corpus.word_embedding, corpus.rela_embedding, q_len, r_len).cuda()
model = HR_BiLSTM(args.hidden_size, corpus.word_embedding, corpus.rela_embedding).cuda()
model.train()
print(model)
if args.optimizer == 'Adadelta':
    optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
elif args.optimizer == 'RMSprop':
    optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
else:
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)


def cal_acc(sorted_score_label):
    if sorted_score_label[0][1] == 1:
        return 1
    else:
        return 0

def save_best_model(model):
    import datetime
    now = datetime.datetime.now()
    if args.save_model_path == '':
        args.save_model_path = f'save_model/{now.month}{now.day}_{now.hour}h{now.minute}m.pt'
    print('save model at {}'.format(args.save_model_path))
    with open(args.save_model_path, 'wb') as outfile:
        torch.save(model, outfile)

def train():
    best_model = None
    best_val_loss = None
    train_start_time = time.time()

    earlystop_counter = 0

    for epoch_count in range(0, args.epoch_num):
        total_loss, total_acc = 0.0, 0.0
        epoch_start_time = time.time()

        for batch_count, batch_data in enumerate(batch_train_data, 1):
            variable_start_time = time.time()
            #training_objs = [obj for q_obj in batch_data for obj in q_obj]
            #question, pos_relas, pos_words, neg_relas, neg_words = zip(*training_objs)
            question, pos_relas, pos_words, neg_relas, neg_words = zip(*batch_data)
            q = Variable(torch.LongTensor(question)).cuda()
            p_relas = Variable(torch.LongTensor(pos_relas)).cuda()
            p_words = Variable(torch.LongTensor(pos_words)).cuda()
            n_relas = Variable(torch.LongTensor(neg_relas)).cuda()
            n_words = Variable(torch.LongTensor(neg_words)).cuda()
            ones = Variable(torch.ones(len(question))).cuda()
            variable_end_time = time.time()
            
            optimizer.zero_grad()
            all_pos_score = model(q, p_relas, p_words)
            all_neg_score = model(q, n_relas, n_words)
            model_end_time = time.time()

            loss = loss_function(all_pos_score, all_neg_score, ones)
            loss.backward()
            optimizer.step()
            loss_backward_time = time.time()
            total_loss += loss.data.cpu().numpy()[0]
            average_loss = total_loss / batch_count

            # Calculate accuracy and f1
            #all_pos = all_pos_score.data.cpu().numpy()
            #all_neg = all_neg_score.data.cpu().numpy()
            #start, end = 0, 0
            #for idx, q_obj in enumerate(batch_data):
            #    end += len(q_obj)
            #    score_list = [all_pos[start]]
            #    #print('len(score_list), score_list')
            #    #print(len(score_list), score_list)
            #    batch_neg_score = all_neg[start:end]
            #    start = end
            #    label_list = [1]
            #    for ns in batch_neg_score:
            #        score_list.append(ns)
            #    label_list += [0] * len(batch_neg_score)
            #    #print('len(score_list), score_list')
            #    #print(len(score_list), score_list)
            #    #print('len(label_list), label_list')
            #    #print(len(label_list), label_list)
            #    score_label = [(x, y) for x, y in zip(score_list, label_list)]
            #    sorted_score_label = sorted(score_label, key=lambda x:x[0], reverse=True)
            #    total_acc += cal_acc(sorted_score_label)
            #average_acc = total_acc / (batch_count * args.batch_size)

            #print(f'variable time      :{variable_end_time-variable_start_time:.3f} / {variable_end_time-epoch_start_time:.3f}')
            #print(f'model time         :{model_end_time - variable_end_time:.3f} / {model_end_time-epoch_start_time:.3f}')
            #print(f'loss calculate time:{loss_backward_time-model_end_time:.3f} / {loss_backward_time-epoch_start_time:.3f}')

            #writer.add_scalar('data/pre_gen_loss', loss.data[0], global_step)
            elapsed = time.time() - epoch_start_time
            print_str = f'Epoch {epoch_count} batch_num:{batch_count} time_elapsed:{elapsed:.2f}s loss:{average_loss*1000:.4f}'
            print('\r', print_str, end='')
            #batch_end_time = time.time()
            #print('one batch', batch_end_time-batch_start_time)
        val_print_str, val_loss, _ = evaluation(model, 'dev')
        print('\r', print_str, 'Val', val_print_str, end='')
        print()

        # this section handle earlystopping
        if not best_val_loss or val_loss < best_val_loss:
            earlystop_counter = 0
            best_model = model
            save_best_model(best_model)
            best_val_loss = val_loss
        else:
            earlystop_counter += 1
        if earlystop_counter >= args.earlystop_tolerance:
            print('EarlyStopping!')
            print(f'Total training time {time.time()-train_start_time:.2f}')
            break
    return best_model

def evaluation(model, mode='dev'):
    model_test = model.eval()
    start_time = time.time()
    total_loss, total_acc = 0.0, 0.0
    if mode == 'test':
        input_data = test_data
    else:
        input_data = val_data
    nb_question = sum(len(batch_data) for batch_data in input_data)
    print('nb_question', nb_question)
    
    for batch_count, batch_data in enumerate(input_data, 1):
        training_objs = [obj for q_obj in batch_data for obj in q_obj]
        question, pos_relas, pos_words, neg_relas, neg_words = zip(*training_objs)
        q = Variable(torch.LongTensor(question)).cuda()
        p_relas = Variable(torch.LongTensor(pos_relas)).cuda()
        p_words = Variable(torch.LongTensor(pos_words)).cuda()
        n_relas = Variable(torch.LongTensor(neg_relas)).cuda()
        n_words = Variable(torch.LongTensor(neg_words)).cuda()
        ones = Variable(torch.ones(len(question))).cuda()
        
        pos_score = model_test(q, p_relas, p_words)
        neg_score = model_test(q, n_relas, n_words)
        loss = loss_function(pos_score, neg_score, ones)
        total_loss += loss.data.cpu().numpy()[0]
        average_loss = total_loss / batch_count

        # Calculate accuracy and f1
        all_pos = pos_score.data.cpu().numpy()
        all_neg = neg_score.data.cpu().numpy()
        start, end = 0, 0
        for idx, q_obj in enumerate(batch_data):
            end += len(q_obj)
            #print('start', start, 'end', end)
            score_list = [all_pos[start]]
            label_list = [1]
            batch_neg_score = all_neg[start:end]
            for ns in batch_neg_score:
                score_list.append(ns)
            label_list += [0] * len(batch_neg_score)
            start = end
            score_label = [(x, y) for x, y in zip(score_list, label_list)]
            #print('len(score_list)', len(score_list), 'len(label_list)', len(label_list), 'len(score_label)', len(score_label))
            sorted_score_label = sorted(score_label, key=lambda x:x[0], reverse=True)
            #print(sorted_score_label)
            total_acc += cal_acc(sorted_score_label)
            #print(total_acc)
            #input('Enter')

        acc1 = total_acc / (batch_count * args.batch_size)
#        acc2 = total_acc / question_counter

    time_elapsed = time.time()-start_time
    average_acc = total_acc / nb_question
    print('acc1', acc1)
#    print('acc2', acc2)
    print('average_acc', average_acc)
#    print(question_counter, nb_question)
    print_str = f'batch_num:{batch_count} time_elapsed:{time_elapsed:.1f}s eval_loss:{average_loss*1000:.4f} eval_acc:{average_acc:.4f}'
    return print_str, average_loss, average_acc

if __name__ == '__main__':
    # Create SummaryWriter
    #writer = SummaryWriter(log_dir=os.path.join(args.save_path, timestep, 'log'))

#    train()
    if args.pretrain_model == None:
        print('Load best model', args.save_model_path)
        with open(args.save_model_path, 'rb') as infile:
            model = torch.load(infile)
    else:
        print('Load pretrain model', args.pretrain_model)
        with open(args.pretrain_model, 'rb') as infile:
            model = torch.load(infile) 
    log_str, _, test_acc = evaluation(model, 'test')
    print(log_str)
    print(test_acc)
    # Close writer
    #writer.close()

