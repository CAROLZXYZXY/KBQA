import json

with open('test_w_r.json', 'r') as f:
    data = json.load(f)

p_relation = []

IC = []

for s in data['Questions']:
    if s['Parses'][0]['InferentialChain'] != None: 
        p_relation.append(s['PotentialRelation'])
        IC.append(s['Parses'][0]['InferentialChain'])

labels = []

for l in IC:
    if len(l) == 1:
        labels.append(l[0])
    elif len(l) == 2:
        labels.append(l[0] + ' ' + l[1])

for index, l in enumerate(labels):
    if l not in p_relation[index]:
        print (index)

