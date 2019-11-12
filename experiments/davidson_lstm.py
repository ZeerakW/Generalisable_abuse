import os
import sys
import torch.nn as nn
import torch.optim as optim
from torchtext.data import TabularDataset, BucketIterator, Field
sys.path.extend(['/Users/zeerakw/Documents/PhD/projects/Generalisable_abuse'])

from gen.shared.data import OnehotBatchGenerator
from gen.neural import LSTMClassifier
from gen.shared.clean import Cleaner
from gen.shared.train import compute_unigram_liwc, train

text_label = Field(sequential = True,
                   include_lengths = False,
                   use_vocab = True,
                   pad_token = "<PAD>",
                   unk_token = "<UNK>")

int_label = Field(sequential = False,
                  include_lengths = False,
                  use_vocab = False,
                  pad_token = None,
                  unk_token = None)

device = 'cpu'
data_dir = '/Users/zeerakw/Documents/PhD/projects/Generalisable_abuse/data/'
# data_file = 'davidson_offensive.csv'
data_file = 'davidson_test.csv'
path = os.path.join(data_dir, data_file)
file_format = 'csv'
cleaners = ['lower', 'url', 'hashtag', 'username']
clean = Cleaner(cleaners)

# Set fields
text_field = text_label
label_field = int_label

# Update training field
setattr(text_field, 'tokenize', clean.tokenize)
setattr(text_field, 'preprocessing', compute_unigram_liwc)
fields = [('', None), ('CF_count', None), ('hate_speech', None), ('offensive', None), ('neither', None),
          ('label', label_field), ('text', text_field)]

data = TabularDataset(path, format = file_format, fields = fields, skip_header = True)
train_data, test = data.split(split_ratio = 0.8, stratified = True)
text_field.build_vocab(train_data)
VOCAB_SIZE = len(text_field.vocab)
loaded = (train_data, test)

print("Vocab Size", len(text_field.vocab))

batch_sizes = (64, 64)
tmp_train, tmp_test = BucketIterator.splits(loaded, batch_sizes = batch_sizes, sort_key = lambda x: len(x.text),
                                            device = device, shuffle = True, repeat = False)
train_batches = OnehotBatchGenerator(tmp_train, 'text', 'label', VOCAB_SIZE)
test_batches = OnehotBatchGenerator(tmp_test, 'text', 'label', VOCAB_SIZE)

model = LSTMClassifier(len(text_field.vocab), embedding_dim = 128, hidden_dim = 128, no_classes = 3, no_layers = 1)
optimizer = optim.Adam(model.parameters(), lr = 0.01)
loss = nn.NLLLoss()

train(model, 5, train_batches, loss, optimizer)
