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
    if len([w for w in sen if len(w) < 10]) != 0:
        continue

    if len([w for w in sen if len(w) > 11]) == 0:
        continue

    variantes = []

    original = [w[1]+(' ' if 'SpaceAfter' not in w[9] else '') for w in sen]
    
    s = ''.join([w[1]+(' ' if 'SpaceAfter' not in w[9] else '') for w in sen])
    #print(s)
    doc = nlp(s) # rebuilding the sentence from the stored text and sending it to ud pipe
    
    if len(doc) != len(sen):
        continue

    deps = [(t.head.i, t.dep_) for t in doc]
    variantes.append(deps)
    print(*[str(d[0])+':'+d[1] for d in deps], s, sep='\t')

    for i, w in enumerate(sen):
        if len(w) == 12: # this word has variants
            alts = w[-1].split(';') # split them

            for alternate in alts: # for each, create the sentence, parse it and compare to the original
                if alternate == '':
                    continue

                new_sen = original[:i] + [alternate + (' ' if 'SpaceAfter' not in w[9] else '')] + original[i+1:]
                new_sen = ''.join(new_sen)

                doc = nlp(new_sen) # rebuilding the sentence from the stored text and sending it to ud pipe
                if len(doc) != len(sen):
                    continue

                deps = [(t.head.i, t.dep_) for t in doc]
                variantes.append(deps)
                print(*[str(d[0])+':'+d[1] + ('<<<' if d != variantes[0][j] else '') for j,d in enumerate(deps)], new_sen, '#######' if deps != variantes[0] else '', sep='\t')
                
                
    print()
    """
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
    """
#print(uas, las, tot, uas/tot*100, las/tot*100, sep='\t')
