from tqdm import tqdm, trange
from sys import argv
from torch import tensor, nn, device, optim, cat, zeros
from random import shuffle


#load the data
data = [l.strip().split('\t') for l in open(argv[1]) if l[0] != '#']

sentences = [[]]
for l in data:
    if l == ['']:
        sentences.append([])
    elif '-' in l[0]:
        ()
    else:
        sentences[-1].append((l[1], l[3]))

sentences = sentences[:-1]
sentences = [s for s in sentences if len(s) < 50] # only keeps not too too long sentences
print(len(sentences))

# make a split for playing with the data
train = sentences[:5000]
dev = sentences[5000:6000]

# get the tokens and labels
tokens = {}
labels = set()

# we count the frequency of each token in the train set
for sen in train:
    for form, pos in sen:
        try:
            tokens[form] += 1
        except:
            tokens[form] = 1
        
        labels.add(pos)

labels.add('UNK') # in case we have not seen all pos in the training set
labels.add('EOS') # end of sentence, to know we have stop
labels = list(sorted(labels))
iol = {l:i for i, l in enumerate(labels)} # gives an index to each label

# gives an index to frequent tokens
iot = {'UNK':0}   # for unseen and too unfrequent tokens
FREQ = 1 # This could be a parameter
for form, freq in sorted(tokens.items()):
    if freq > FREQ:
        iot[form] = len(iot)
    else:
        # if a token is not frequent enough, we will replace it by UNK
        ()
       
# numerize the data
train = [[(iot[form] if form in iot else iot['UNK'], iol[pos] if pos in iol else iol['UNK']) for form, pos in sen] for sen in train]
dev = [[(iot[form] if form in iot else iot['UNK'], iol[pos] if pos in iol else iol['UNK']) for form, pos in sen] for sen in dev]

train = [[*zip(*sen)] for sen in train]
dev = [[*zip(*sen)] for sen in dev]

# and add the end of sequence symbol to stop generatioN
train = [(sen, pos+(iol['EOS'],)) for sen, pos in train]
dev = [(sen, pos+(iol['EOS'],)) for sen, pos in dev]


# no to the neural bits
device = device('cuda:0') # use device('cpu') if you don't have a GPU

    
# the classes
class Model0(nn.Module):
    """
    not seq2seq, plain vanilla RNN, this means no decoder
    """

    def __init__(self, num_tokens, token_dim, hidden_dim, num_label, device):
        nn.Module.__init__(self)

        self.hidden_dim = hidden_dim
        
        self.E = nn.Embedding(num_tokens, token_dim)
        self.FBencoder = nn.GRU(token_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.last = nn.Linear(2*hidden_dim, num_label)

        self.loss = nn.CrossEntropyLoss()
        
        self.trainer = optim.Adadelta(self.parameters())
        self.device = device
        self.to(device)


    def train(self, x, y):
        self.trainer.zero_grad()

        # embed the token indices
        emb = self.E(tensor(x, device=self.device))
        # feed the whole sentence into a forward/backward encoder
        encoded = self.FBencoder(emb)[0]

        # now predict
        scores = self.last(encoded) # compute the likelihood of each POS given its word's encoding
        loss = self.loss(scores, tensor(y[:-1], device=self.device)) # compute loss
            
        loss.backward()
        self.trainer.step()
            

    def predict(self, x):
        self.trainer.zero_grad()
        
        # embed the token indices
        emb = self.E(tensor(x, device=self.device))
        encoded = self.FBencoder(emb)[0]

        # now predict
        scores = self.last(encoded) # compute the likelihood of each POS given its word's encoding
        pred = scores.argmax(dim=1)

        return pred





class Model1(nn.Module):
    """
    vanilla seq2seq, no attention
    """

    def __init__(self, num_tokens, token_dim, hidden_dim, num_label, device, eos):
        nn.Module.__init__(self)

        self.hidden_dim = hidden_dim
        self.eos = eos
        
        self.E = nn.Embedding(num_tokens, token_dim)
        self.L = nn.Embedding(num_label, hidden_dim)
        self.Fencoder = nn.GRU(token_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.Bencoder = nn.GRU(token_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.middle = nn.Linear(2*hidden_dim, hidden_dim)
        self.length = nn.Linear(hidden_dim, 50)

        self.decoder = nn.GRU(2*hidden_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.last = nn.Linear(hidden_dim, num_label, bias=False)
        self.dropout = nn.Dropout(0.5)
        
        self.loss = nn.CrossEntropyLoss()
        
        self.trainer = optim.Adam(self.parameters())
        self.device = device
        self.to(device)


    def train(self, x, y):
        self.trainer.zero_grad()

        # embed the token indices
        emb = self.E(tensor(x, device=self.device))
        # take the last hidden state of the forward and backward encoders and concat them
        encoded = cat([self.Fencoder(emb)[1], self.Bencoder(emb.flip([0]))[1]], dim=1)
        #print(encoded.size())
        # reduce the size if the sentence representation
        sent = self.middle(encoded)

        lscores = self.length(self.dropout(sent))
        lenloss = self.loss(lscores, tensor([len(x)], device=self.device))
        #print(lscores, lscores.argmax(dim=1), len(x))
        loss = 30 * lenloss # to make sure the length loss is not too diluted in the POS loss
        
        # now decode
        h = cat([zeros((1, self.hidden_dim), device=self.device), sent], dim=1) # initialize decoder with sentence encoding
        for i in range(lscores.argmax(dim=1).item()):
            h = self.decoder(h)[1]
            scores = self.last(h) # compute the likelihood of each POS given the current hidden state
            pred = scores.argmax(dim=1)

            if i >= len(y):
                loss += self.loss(scores, tensor([self.eos], device=self.device)) # compute loss
            else:
                loss += self.loss(scores, tensor([y[i]], device=self.device)) # compute loss

            #print(scores, pred)
            h = cat([h, self.L(pred)], dim=1) # next imput, current hidden state + predicted POS

        loss.backward()
        self.trainer.step()
            

    def predict(self, x):
        self.trainer.zero_grad()
        
        # embed the token indices
        emb = self.E(tensor(x, device=self.device))
        # take the last hidden state of the forward and backward encoders and concat them
        encoded = cat([self.Fencoder(emb)[1], self.Bencoder(emb.flip([0]))[1]], dim=1)
        #print(encoded.size())
        # reduce the size if the sentence representation
        sent = self.middle(encoded)

        lscores = self.length(sent)
        plen = lscores.argmax(dim=1).item()
        
        # now decode
        pred = [] # count the number of generated labels
        
        h = cat([zeros((1, self.hidden_dim), device=self.device), sent], dim=1) # initialize decoder with sentence encoding
        for i in range(plen):
            h = self.decoder(h)[1]
            scores = self.last(h) # compute the likelihood of each POS given the current hidden state
            pred.append(scores.argmax(dim=1).item())
            h = cat([h, self.L(scores.argmax(dim=1))], dim=1) # next imput, current hidden state + predicted POS
        
        return pred, plen






class Model2(nn.Module):

    def __init__(self, num_tokens, token_dim, hidden_dim, num_label, device, eos):
        nn.Module.__init__(self)

        self.hidden_dim = hidden_dim
        self.eos = eos
        
        self.E = nn.Embedding(num_tokens, token_dim)
        self.L = nn.Embedding(num_label, hidden_dim)
        self.FBencoder = nn.GRU(token_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.middle = nn.Linear(2*hidden_dim, hidden_dim)
        self.length = nn.Linear(hidden_dim, 50)

        self.decoder = nn.GRU(4*hidden_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.last = nn.Linear(hidden_dim, num_label)
        self.dropout = nn.Dropout(0.5)
        
        self.att = nn.MultiheadAttention(2*hidden_dim, num_heads=1, batch_first=True)
        
        self.loss = nn.CrossEntropyLoss()
        
        self.trainer = optim.Adam(self.parameters())
        self.device = device
        self.to(device)


    def train(self, x, y):
        self.trainer.zero_grad()

        # embed the token indices
        emb = self.E(tensor(x, device=self.device))
        # take the last hidden state of the forward and backward encoders and concat them
        encoded, hs = self.FBencoder(emb)
        #print(encoded.size(), hs.size())
        #print(encoded[0])
        #print(hs)
        # reduce the size if the sentence representation
        sent = self.middle(cat([encoded[0][100:], encoded[-1][:100]], dim=0).reshape((1, 2*self.hidden_dim)))
        
        lscores = self.length(self.dropout(sent))
        lenloss = self.loss(lscores, tensor([len(x)], device=self.device))
        loss = 30 * lenloss
        
        # now decode
        h = cat([zeros((1, self.hidden_dim), device=self.device), sent], dim=1) # initialize decoder with sentence encoding
        for i in range(lscores.argmax(dim=1).item()):
            att = self.att(h, encoded, encoded)[0]
            h = self.decoder(cat([h, att], dim=1))[1]
            scores = self.last(h) # compute the likelihood of each POS given the current hidden state
            pred = scores.argmax(dim=1)
            
            if i >= len(y):
                loss += self.loss(scores, tensor([self.eos], device=self.device)) # compute loss
            else:
                loss += self.loss(scores, tensor([y[i]], device=self.device)) # compute loss

            h = cat([h, self.L(scores.argmax(dim=1))], dim=1) # next imput, current hidden state + predicted POS
                   
        loss.backward()
        self.trainer.step()
            

    def predict(self, x):
        self.trainer.zero_grad()
        
        # embed the token indices
        emb = self.E(tensor(x, device=self.device))
        # take the last hidden state of the forward and backward encoders and concat them
        encoded = self.FBencoder(emb)[0]
        #print(encoded.size())
        # reduce the size if the sentence representation
        sent = self.middle(cat([encoded[0][100:], encoded[-1][:100]], dim=0).reshape((1, 2*self.hidden_dim)))

        lscores = self.length(sent)
        plen = lscores.argmax(dim=1).item()

        # now decode
        pred = [] # count the number of generated labels
        
        h = cat([zeros((1, self.hidden_dim), device=self.device), sent], dim=1) # initialize decoder with sentence encoding
        for i in range(plen):
            att = self.att(h, encoded, encoded)[0]
            h = self.decoder(cat([h, att], dim=1))[1]
            scores = self.last(h) # compute the likelihood of each POS given the current hidden state
            pred.append(scores.argmax(dim=1).item())
            h = cat([h, self.L(scores.argmax(dim=1))], dim=1) # next imput, current hidden state + predicted POS
        
        return pred, plen








    

# set and train and test
#model = Model0(len(iot), 100, 100, len(iol), device)
#model = Model1(len(iot), 100, 100, len(iol), device, iol['EOS'])
model = Model2(len(iot), 100, 100, len(iol), device, iol['EOS'])

# note how model0 is much better than 1 and 2, but given enough data and time and thought, it would work, and better, it works for other tasks than sequence label

for _ in trange(5):
    shuffle(train)
    for toks, labels in tqdm(train, leave=False):
        model.train(toks, labels)

    # after each epoch test on train, and on test,   train is used a very simple sanity check
    good = 0
    tot = 0
    toolong = 0
    tooshort = 0
    lgood = 0
    for toks, labels in tqdm(train, leave=False):
        pred, ln = model.predict(toks)
        #print(pred)
        #print(labels)
        #print()
        if ln == len(toks):
            lgood += 1
        
        for i, l in enumerate(labels):
            tot += 1
            if i >= len(pred):
                tooshort += 1
            elif l == pred[i]:
                good += 1
            
        toolong += max(0, len(pred)-len(toks)-1)

    print(' Train : ', good, tot, good/tot*100, toolong, tooshort, lgood, lgood/len(train)*100, sep='\t')


    good = 0
    tot = 0
    toolong = 0
    tooshort = 0
    lgood = 0
    for toks, labels in tqdm(dev, leave=False):
        pred, ln = model.predict(toks)
        #print(pred)
        #print(labels)
        #print()
        if ln == len(toks):
            lgood += 1

        for i, l in enumerate(labels):
            tot += 1
            if i >= len(pred):
                tooshort += 1
            elif l == pred[i]:
                good += 1
            
        toolong += max(0, len(pred)-len(toks)-1)

    print(' Test : ', good, tot, good/tot*100, toolong, tooshort, lgood, lgood/len(dev)*100, sep='\t')
