import os
import sys
import pdb
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchtext.data import TabularDataset, BucketIterator
sys.path.extend(['/Users/zeerakw/Documents/PhD/projects/Generalisable_abuse'])

import gen.shared.types as t
from gen.shared.clean import Cleaner
from gen.shared.data import BatchGenerator
from gen.shared.train import compute_unigram_liwc
from gen.neural import LSTMClassifier

device = 'cpu'
data_dir = '/Users/zeerakw/Documents/PhD/projects/Generalisable_abuse/data/'
data_file = 'davidson_test.csv'
path = os.path.join(data_dir, data_file)
file_format = 'csv'
cleaners = ['lower', 'url', 'hashtag', 'username']
clean = Cleaner(cleaners)

# Set fields
text_field = t.text_data
label_field = t.int_label

# Update training field
setattr(text_field, 'tokenize', clean.tokenize)
setattr(text_field, 'preprocessing', compute_unigram_liwc)

fields = [('', None), ('CF_count', None), ('hate_speech', None), ('offensive', None), ('neither', None),
          ('label', label_field), ('text', text_field)]

data = TabularDataset(path, format = file_format, fields = fields, skip_header = True)
train, test = data.split(split_ratio = 0.8, stratified = True)
loaded = (train, test)
batch_sizes = (64, 64)

text_field.build_vocab(train)
label_field.build_vocab(train)

tmp_train, tmp_test = BucketIterator.splits(loaded, batch_sizes = batch_sizes, sort_key = lambda x: len(x.data),
                                            device = device, shuffle = True, repeat = True)


batched_train = BatchGenerator(tmp_train, 'text', 'label')
# batched_test = BatchGenerator(tmp_test, 'data', 'label')

# Model options
HIDDEN_DIM = 300
NUM_CLASSES = 3
NUM_LAYERS = 2
VOCAB_SIZE = len(text_field.vocab)

model = LSTMClassifier(VOCAB_SIZE, HIDDEN_DIM, NUM_CLASSES, NUM_LAYERS)
optimizer = optim.Adam(model.parameters(), lr = 0.01)
loss = nn.NLLLoss()


def train_model(model, epochs, batches, loss_func, optimizer):

    for batch in tqdm(tmp_train):
        model.zero_grad()
        pdb.set_trace()
        scores = model(batch.text)
        loss = loss_func(scores, dataY)
        loss.backward()
        optimizer.step()


train_model(model, 300, tmp_train, loss, optimizer)
