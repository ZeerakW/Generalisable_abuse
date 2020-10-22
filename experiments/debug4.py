import wandb
import torch
import numpy as np
from tqdm import tqdm, trange
from mlearn.utils.metrics import Metrics
from mlearn.data.clean import Cleaner, Preprocessors
from mlearn.utils.train import train_singletask_model
from argparse import ArgumentParser
from jsonargparse import ArgumentParser, ActionConfigFile
from mlearn.utils.train import run_singletask_model as run_model
from mlearn.modeling.embedding import MLPClassifier, CNNClassifier
from mlearn.utils.pipeline import process_and_batch, param_selection
from mlearn.data.batching import TorchtextExtractor
from torchtext.data import TabularDataset, Field, LabelField, BucketIterator

if __name__ == "__main__":
   parser = ArgumentParser(description = "Run Experiments using MTL.")

   # Data inputs and outputs
   parser.add_argument("--main", help = "Choose train data: Davidson, Waseem, Waseem and Hovy, Wulczyn, and Garcia.",
                       type = str.lower, default = 'Davidson')
   parser.add_argument("--aux", help = "Specify the auxiliary datasets to be loaded.", type = str, nargs = '+')
   parser.add_argument("--datadir", help = "Path to the datasets.", default = 'data/')
   parser.add_argument("--results", help = "Set file to output results to.", default = 'results/')
   parser.add_argument("--save_model", help = "Directory to store models in.", default = 'results/models/')

   # Cleaning and metrics
   parser.add_argument("--cleaners", help = "Set the cleaning routines to be used.", nargs = '+', default = None)
   parser.add_argument("--metrics", help = "Set the metrics to be used.", nargs = '+', default = ["f1"],
                       type = str.lower)
   parser.add_argument("--display", help = "Metric to display in TQDM loops.", default = 'f1-score')
   parser.add_argument("--stop_metric", help = "Set the metric to be used for early stopping", default = "loss")

   # Experiment
   parser.add_argument('--tokenizer', help = "select the tokenizer to be used: Spacy, Ekphrasis, BPE",
                       default = 'ekphrasis', type = str.lower)
   parser.add_argument("--experiment", help = "Set experiment to run.", default = "word", type = str.lower)

   # Modelling
   ## All models
   parser.add_argument("--patience", help = "Set the number of epochs to keep trying to find a new best",
                       default = None, type = int)
   parser.add_argument("--model", help = "Choose the model to be run: CNN, RNN, LSTM, MLP, LR.", nargs = '+',
                       default = ['mlp'], type = str.lower)
   parser.add_argument('--encoding', help = "Select encoding to be used: Onehot, Index, Tfidf, Count",
                       default = 'index', type = str.lower)
   parser.add_argument("--optimizer", help = "Optimizer to use.", default = 'adam', type = str.lower)
   parser.add_argument("--loss", help = "Loss to use.", default = 'nlll', type = str.lower)
   parser.add_argument('--seed', help = "Set the random seed.", type = int, default = 32)
   parser.add_argument('--shuffle', help = "Shuffle dataset between epochs", type = bool, default = True)
   parser.add_argument('--gpu', help = "Set to run on GPU", type = int, default = 0)
   ## LSTM
   parser.add_argument("--layers", help = "Set the number of layers.", default = 1, type = int)

   # Hyper Parameters
   parser.add_argument('--window_sizes', help = "Set CNN window sizes.", default = ["2,3,4"],
                       type = str, nargs = '+')
   parser.add_argument("--filters", help = "Set the number of filters for CNN.", default = [128],
                       type = int,  nargs = '+')
   parser.add_argument("--embedding", help = "Set the embedding dimension.", default = [100],
                       type = int,  nargs = '+')
   parser.add_argument("--hidden", help = "Set the hidden dimension.", default = [128], type = int,
                       nargs = '+')
   parser.add_argument("--shared", help = "Set the shared dimension", default = [256], type = int,
                       nargs = '+')
   parser.add_argument("--epochs", help = "Set the number of epochs.", default = [200], type = int,
                       nargs = '+')
   parser.add_argument("--batch_size", help = "Set the batch size.", default = [64], type = int,
                       nargs = '+')
   parser.add_argument("--nonlinearity", help = "Set nonlinearity function for neural nets.",
                       default = ['tanh'], type = str.lower, nargs = '+')
   parser.add_argument('--learning_rate', help = "Set the upper limit for the learning rate.",
                       default = 1.0, type = float)
   parser.add_argument("--dropout",  help = "Set upper limit for dropout.", default = 1.0, type = float)

   # args
   # parser.add_argument('--cfg', action = ActionConfigFile, default = None)
   # parser.add_argument('--wandb', action = ActionConfigFile, default = None)
   args = parser.parse_args()

   # wandb_vals = dict(window_sizes = args.window_sizes,
   #                   filters = args.filters,
   #                   embedding_dims = args.embedding,
   #                   hidden_dims = args.hidden,
   #                   shared_dims = args.shared,
   #                   epochs = args.epochs,
   #                   batch_size = args.batch_size,
   #                   nonlinearity = args.nonlinearity,
   #                   learning_rate = args.learning_rate,
   #                   dropout = args.dropout)

   if 'f1' in args.metrics:
      args.metrics[args.metrics.index('f1')] = 'f1-score'
   if args.display == 'f1':
       args.display = 'f1-score'
   if args.stop_metric == 'f1':
       args.stop_metric= 'f1-score'

   wandb.init('test', project = 'generalise', entity = 'zeerak')
   config = wandb.config
   # config.update(wandb_vals)  # Update

   datadir = 'data/json'
   torch.random.manual_seed(42)
   np.random.seed(42)
   metrics = ['f1-score', 'precision', 'recall', 'accuracy']
   display_metric = stop_metric = 'f1-score'
   filters = 128
   window_sizes = [2,3,4]
   gpu = args.gpu
   hyperopt = False

   c = Cleaner(['url', 'hashtag', 'username', 'lower'])
   experiment = Preprocessors('data/').select_experiment(args.experiment)
   onehot = True if args.encoding == 'onehot' else False

   if args.tokenizer == 'spacy':
       selected_tok  = c.tokenize
   elif args.tokenizer == 'bpe':
       selected_tok = c.bpe_tokenize
   elif args.tokenizer == 'ekphrasis' and exp == 'word':
       selected_tok = c.ekphrasis_tokenize
       annotate = {'elongated', 'emphasis'}
       flters = [f"<{filtr}>" for filtr in annotate]
       c._load_ekphrasis(annotate, flters)
   elif args.tokenizer == 'ekphrasis' and exp == 'liwc':
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

   if args.main == 'davidson':
       train, dev, test = TabularDataset.splits(args.datadir, train = 'davidson_binary_train.json',
                                                validation = 'davidson_binary_dev.json',
                                                test = 'davidson_binary_test.json', 
                                                format = 'json', skip_header = True, fields = fields)
   elif args.main == 'wulczyn':
       train, dev, test = TabularDataset.splits(args.datadir, train = 'wulczyn_train.json',
               validation = 'wulczyn_dev.json',
               test = 'wulczyn_test.json', 
               format = 'json', skip_header = True, fields = fields)
   elif args.main == 'wasem':
       train, dev, test = TabularDataset.splits(args.datadir, train = 'waseem_train.json',
               validation = 'waseem_dev.json',
               test = 'waseem_test.json', 
               format = 'json', skip_header = True, fields = fields)
   text.build_vocab(train)
   label.build_vocab(train)

   train_metrics = Metrics(metrics, display_metric, stop_metric)
   dev_metrics = Metrics(metrics, display_metric, stop_metric)

   # model = MLPClassifier(len(text.vocab.stoi), config.embedding, config.hidden, len(label.vocab.stoi), 
   #                       config.dropout, True, config.nonlinearity)
   model = CNNClassifier([int(win) for win in config.window_sizes.split(',')], config.filters, len(text.vocab.stoi),
                          config.embedding, len(label.vocab.stoi), config.nonlinearity, True)
   loss = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.SGD(model.parameters(), config.learning_rate)

   train_ds = BucketIterator(dataset = train, batch_size = config.batch_size)
   dev_ds = BucketIterator(dataset = dev, batch_size = 64)
   batched_train = TorchtextExtractor('text', 'label', 'davidson_binary_train', train_ds)
   batched_dev = TorchtextExtractor('text', 'label', 'davidson_binary_dev', dev_ds)

   train_singletask_model(model, 'results/models/', config.epochs, batched_train, loss, optimizer, train_metrics,
                          dev = batched_dev, dev_metrics = dev_metrics, shuffle = False, gpu = gpu, hyperopt = True,
                          clip = 1.0, early_stopping = 10)

   # for key in train_metrics.scores:
   #     for val in train_metrics.scores[key]:
   #         wandb.log({key: val})
   #     for val in dev_metrics.scores[key]:
   #         wandb.log({f'dev_{key}': val})

   arg = np.argmax(train_metrics.scores['f1-score'])
   print(arg, train_metrics.scores['f1-score'][arg])

   arg = np.argmax(dev_metrics.scores['f1-score'])
   print(arg, dev_metrics.scores['f1-score'][arg])

