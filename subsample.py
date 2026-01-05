from sys import argv
from tqdm import tqdm
from collections import defaultdict
from random import seed, shuffle
import spacy_udpipe

seed(0) # set a seed for reproducibility

nlp = spacy_udpipe.load('fr') # load the model

# load the treebank
data = []
sen = [], []

for l in tqdm(open(argv[1])):
    l = l.strip()

    if l == '':
        data.append(sen)
        sen = [], []
        continue
    
    if l[0] == '#':
        sen[0].append(l)

    else:
        l = l.split('\t')
        if '-' in l[0]:
            continue
        sen[1].append(l)

# sort the data and prepare for subsampling
length = defaultdict(list)
for h, sen in tqdm(data):

    k = len(sen)
    if (k > 3 and k < 11) or (k % 5 == 0 and k < 75): 
        s = ''.join([w[1]+(' ' if 'SpaceAfter' not in w[-1] else '') for w in sen])
        #print(s)
        doc = nlp(s) # rebuilding the sentence from the stored text and sending it to ud pipe

        if [t.text for t in doc] == [w[1] for w in sen]:
            length[k].append((h, sen))


for k, sens in sorted(length.items()):
    
    if (k > 2 and k < 11) or (k % 5 == 0 and k < 75):
        shuffle(sens)
        for hs, ws in sens[:10]:
            for h in hs:
                print(h)
            for w in ws:
                print('\t'.join(w))
            print()
            
            #print(k, v, sep='\t')
            
