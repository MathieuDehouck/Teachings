import spacy_udpipe

from sys import argv
from tqdm import tqdm # not strictly necessary

#spacy_udpipe.download('fr')  # you may need to download the model

nlp = spacy_udpipe.load('fr') # load the model

# read the data
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


# test the model

uas = 0
las = 0
tot = 0
for head, sen in tqdm(data):
    s = ''.join([w[1]+(' ' if 'SpaceAfter' not in w[-1] else '') for w in sen])
    #print(s)
    doc = nlp(s) # rebuilding the sentence from the stored text and sending it to ud pipe
    
    #for token in doc:
    #print(token.text, token.lemma_, token.pos_, token.dep_)

    if len(doc) != len(sen):
        # some tokenization issues maybe
        #print(len(doc), [t for t in doc])
        #print(' '.join([w[1] for w in sen]))
        continue
        
    for w, tok in zip(sen, doc):
        tot += 1

        if w[6] == '0': # spacy ne gère pas les racines comme UD
            if tok.head.i == tok.i:
                uas += 1
                las += 1

        elif tok.head.i == int(w[6])-1:
            uas += 1
            if tok.dep_ == w[7]:
                las += 1

    #print()

print(uas, las, tot, uas/tot*100, las/tot*100, sep='\t')
