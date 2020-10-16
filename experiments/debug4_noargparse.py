import wandb
import torch
import numpy as np
from mlearn import base
from tqdm import tqdm, trange
from mlearn.utils.metrics import Metrics
from mlearn.data.batching import TorchtextExtractor
from mlearn.data.clean import Cleaner, Preprocessors
from mlearn.utils.train import train_singletask_model
from mlearn.utils.train import run_singletask_model as run_model
from mlearn.modeling.embedding import MLPClassifier, CNNClassifier
from mlearn.utils.pipeline import process_and_batch, param_selection
from torchtext.data import TabularDataset, Field, LabelField, BucketIterator

val = {'epochs': 100, 'learning_rate': 0.4653820065445793, 'nonlinearity': 'relu', 'batch_size': 64,
      'hidden': 300, 'dropout': 0.03623935933654854, 'embedding': 200}
wandb.init('test', project = 'Generalise-experiments', entity = 'zeerak')
config = wandb.config
config.update(val)

datadir = 'data/json'
main = 'davidson'
torch.random.manual_seed(42)
np.random.seed(42)
metrics = ['f1-score', 'precision', 'recall', 'accuracy']
display_metric = stop_metric = 'f1-score'
gpu = True
hyperopt = True

c = Cleaner(['url', 'hashtag', 'username', 'lower'])
exp = 'word'
experiment = Preprocessors('data/').select_experiment('word')
onehot = False
tokenizer = 'ekphrasis'

if tokenizer == 'spacy':
   selected_tok  = c.tokenize
elif tokenizer == 'bpe':
   selected_tok = c.bpe_tokenize
elif tokenizer == 'ekphrasis' and exp == 'word':
   selected_tok = c.ekphrasis_tokenize
   annotate = {'elongated', 'emphasis'}
   flters = [f"<{filtr}>" for filtr in annotate]
   c._load_ekphrasis(annotate, flters)
elif tokenizer == 'ekphrasis' and exp == 'liwc':
   ekphr = c.ekphrasis_tokenize
   annotate = {'elongated', 'emphasis'}
   flters = [f"<{filtr}>" for filtr in annotate]
   c._load_ekphrasis(annotate, flters)

   def liwc_toks(doc):
       tokens = ekphr(doc)
       tokens = experiment(tokens)
       return tokens
   selected_tok = liwc_toks

tokenizer = selected_tok
text = Field(tokenize = tokenizer, lower = True, batch_first = True)
label = LabelField()
fields = {'text': ('text', text), 'label': ('label', label)}  # Because we load from json we just need this.

if main == 'davidson':
   train, dev, test = TabularDataset.splits(datadir, train = 'davidson_binary_train.json',
                                            validation = 'davidson_binary_dev.json',
                                            test = 'davidson_binary_test.json', 
                                            format = 'json', skip_header = True, fields = fields)
elif main == 'wulczyn':
   train, dev, test = TabularDataset.splits(datadir, train = 'wulczyn_train.json',
           validation = 'wulczyn_dev.json',
           test = 'wulczyn_test.json', 
           format = 'json', skip_header = True, fields = fields)
elif main == 'wasem':
   train, dev, test = TabularDataset.splits(datadir, train = 'waseem_train.json',
           validation = 'waseem_dev.json',
           test = 'waseem_test.json', 
           format = 'json', skip_header = True, fields = fields)
text.build_vocab(train)
label.build_vocab(train)

train_metrics = Metrics(metrics, display_metric, stop_metric)
dev_metrics = Metrics(metrics, display_metric, stop_metric)

model = MLPClassifier(len(text.vocab.stoi), config.embedding, config.hidden, len(label.vocab.stoi), 
                     config.dropout, True, config.nonlinearity)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), config.learning_rate)

train_ds = BucketIterator(dataset = train, batch_size = config.batch_size)
dev_ds = BucketIterator(dataset = dev, batch_size = 64)
batched_train = TorchtextExtractor('text', 'label', 'davidson_binary_train', train_ds)
batched_dev = TorchtextExtractor('text', 'label', 'davidson_binary_dev', dev_ds)

breakpoint()
train_singletask_model(model, 'data/testing_wandb', config.epochs, batched_train, loss, optimizer, train_metrics,
                      dev = batched_dev, dev_metrics = dev_metrics, shuffle = False, gpu = True,
                      clip = 1.0, early_stopping = 10, hyperopt = hyperopt)

arg = np.argmax(train_metrics.scores['f1-score'])
print(arg, train_metrics.scores['f1-score'][arg])

arg = np.argmax(dev_metrics.scores['f1-score'])
print(arg, dev_metrics.scores['f1-score'][arg])
