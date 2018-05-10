''' Split WebQSP.all (WebQSP.train + WebQSP.test from KBQA_RE_data) 
    to training and testing according to hops of answer relation
'''
import sys

if __name__ == '__main__':
    relation_path = 'webqsp_relations.txt'
    training_data_path = 'WebQSP'
    relations_dic = {}

    with open(relation_path) as infile:
        idx = 1
        for line in infile:
            relations_dic[str(idx)] = line.strip()
            idx += 1

    onehop = []
    twohop = []
    undefined = []
    with open(training_data_path+'.all') as infile:
        for idx, line in enumerate(infile, 1):
            print('\r', idx, end='')
            tokens = line.strip().split('\t')
            pos_relations = tokens[0]
            hops = 0
            for pos_id in pos_relations.split(' '):
                if hops == 0:
                    hops = len(relations_dic[pos_id].split('..'))
                else:
                    if len(relations_dic[pos_id].split('..')) != hops:
                        undefined.append(line)
                        hops = -1
                        break
            if hops == 1:
                onehop.append(line)
            elif hops == 2:
                twohop.append(line)
            elif hops == -1:
                continue
            else:
                print(line)
                print('nb_pos', len(pos_relations.split(' ')))
                print('hops', hops)
        print()
 
    print(len(onehop), len(twohop), len(undefined))                

    with open(training_data_path+'.1hop', 'w') as outfile:
        for line in onehop:
            outfile.write(line)
    
    with open(training_data_path+'.2hop', 'w') as outfile:
        for line in twohop:
            outfile.write(line)

    with open(training_data_path+'.undefined', 'w') as outfile:
        for line in undefined:
            outfile.write(line)
    
