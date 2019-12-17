import os
import sys
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field
from torchtext.data import TabularDataset, BucketIterator, Iterator
sys.path.extend(['/Users/zeerakw/Documents/PhD/projects/active/Generalisable_abuse'])

from gen.neural import MLPClassifier
from gen.shared.clean import Cleaner
from gen.shared.data import OnehotBatchGenerator
from gen.shared.train import train_model, evaluate_model
from sklearn.metrics import accuracy_score

text_field = Field(sequential = True,
                   include_lengths = False,
                   use_vocab = True,
                   pad_token = "<PAD>",
                   unk_token = "<UNK>")

int_field = Field(sequential = False,
                  include_lengths = False,
                  use_vocab = False,
                  pad_token = None,
                  unk_token = None)

cleaners = ['lower', 'url', 'hashtag', 'username']
clean = Cleaner(cleaners)

# Update training field
setattr(text_field, 'tokenize', clean.tokenize)
# setattr(text_field, 'preprocessing', clean.compute_unigram_liwc)
# setattr(text_field, 'preprocessing', clean.ptb_tokenize)

fields = [('', None), ('CF_count', None), ('hate_speech', None), ('offensive', None), ('neither', None),
          ('label', int_field), ('text', text_field)]
text_field = text_field
label_field = int_field
batch_sizes = (32, 32)
path = '/Users/zeerakw/Documents/PhD/projects/active/Generalisable_abuse/data/'
train_file = 'davidson_train.csv'
test_file = 'davidson_test.csv'

train = TabularDataset(os.path.join(path, train_file), format = 'csv', fields = fields, skip_header = True)
text_field.build_vocab(train)
test  = TabularDataset(os.path.join(path, test_file), format = 'csv', fields = fields, skip_header = True)
train, dev = train.split(split_ratio = 0.8, stratified = True)

loaded = (train, dev)
VOCAB_SIZE = len(text_field.vocab)

print("Vocab Size", len(text_field.vocab))

train_batch, dev_batch = BucketIterator.splits(loaded, batch_sizes = batch_sizes, sort_key = lambda x: len(x.text),
                                               device = 'cpu', shuffle = True, repeat = False)
test_batch = Iterator(test, batch_size = 32, sort = False, sort_within_batch = False, repeat = False)

train_batches = OnehotBatchGenerator(train_batch, 'text', 'label', VOCAB_SIZE)
dev_batches = OnehotBatchGenerator(dev_batch, 'text', 'label', VOCAB_SIZE)
test_batches = OnehotBatchGenerator(test_batch, 'text', 'label', VOCAB_SIZE)

model = MLPClassifier(VOCAB_SIZE, hidden_dim = 128, output_dim = 3)
optimizer = optim.Adam(model.parameters(), lr = 0.01)
loss = nn.NLLLoss()

# train_model(model, 5, train, dev, loss, optimizer, text_field)
train_model(model, 5, train_batches, loss, optimizer, text_field)
evaluate_model(model, dev_batches, loss, accuracy_score, "accuracy")
evaluate_model(model, test_batches, loss, accuracy_score, "accuracy")
