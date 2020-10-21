import os
import csv
import json
import wandb
import torch
import numpy as np
from tqdm import tqdm
from mlearn import base
from argparse import ArgumentParser
from mlearn.utils.metrics import Metrics
from mlearn.modeling import onehot as oh
from mlearn.modeling import embedding as emb
from mlearn.data.dataset import GeneralDataset
from mlearn.utils.pipeline import process_and_batch
from mlearn.data.batching import TorchtextExtractor
from mlearn.data.clean import Cleaner, Preprocessors
from mlearn.utils.train import run_singletask_model
from torchtext.data import TabularDataset, Field, LabelField, BucketIterator


if __name__ == "__main__":
    parser = ArgumentParser(description = "Run Experiments using MTL.")

    # Data inputs and outputs
    parser.add_argument("--main", help = "Choose train data: Davidson, Waseem, Waseem and Hovy, Wulczyn, and Garcia.",
                        type = str.lower, default = 'Davidson')
    parser.add_argument("--aux", help = "Specify the auxiliary datasets to be loaded.", type = str, nargs = '+', default = ['waseem'])
    parser.add_argument("--datadir", help = "Path to the datasets.", default = 'data/json/')
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
    # All models
    parser.add_argument("--patience", help = "Set the number of epochs to keep trying to find a new best",
                        default = 15, type = int)
    parser.add_argument("--model", help = "Choose the model to be run: CNN, RNN, LSTM, MLP, LR.", default = 'mlp',
                        type = str.lower)
    parser.add_argument('--encoding', help = "Select encoding to be used: Onehot, Index, Tfidf, Count",
                        default = 'index', type = str.lower)
    parser.add_argument("--optimizer", help = "Optimizer to use.", default = 'adam', type = str.lower)
    parser.add_argument("--loss", help = "Loss to use.", default = 'nlll', type = str.lower)
    parser.add_argument('--seed', help = "Set the random seed.", type = int, default = 32)
    parser.add_argument('--shuffle', help = "Shuffle dataset between epochs", type = bool, default = True)
    parser.add_argument('--gpu', help = "Set to run on GPU", type = int, default = 0)
    # LSTM
    parser.add_argument("--layers", help = "Set the number of layers.", default = 1, type = int)

    # Hyper Parameters
    parser.add_argument("--embedding", help = "Set the embedding dimension.", default = 100, type = int)
    parser.add_argument("--hidden", help = "Set the hidden dimension.", default = 128, type = int)
    parser.add_argument("--epochs", help = "Set the number of epochs.", default = 200, type = int)
    parser.add_argument("--batch_size", help = "Set the batch size.", default = 64, type = int)
    parser.add_argument("--nonlinearity", help = "Set nonlinearity function for neural nets.",
                        default = 'tanh', type = str.lower)
    parser.add_argument('--learning_rate', help = "Set the upper limit for the learning rate.",
                        default = 1.0, type = float)
    parser.add_argument("--dropout", help = "Set upper limit for dropout.", default = 1.0, type = float)
    # CNN
    parser.add_argument('--window_sizes', help = "Set CNN window sizes.", default = "2,3,4",
                        type = str)
    parser.add_argument("--filters", help = "Set the number of filters for CNN.", default = 128,
                        type = int)
    args = parser.parse_args()

    if 'f1' in args.metrics:
        args.metrics[args.metrics.index('f1')] = 'f1-score'
    if args.display == 'f1':
        args.display = 'f1-score'
    if args.stop_metric == 'f1':
        args.stop_metric = 'f1-score'

    # Set up WandB logging
    wandb.init(config = args)
    config = wandb.config

    # Initialise random seeds
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set up experiment and cleaner
    c = Cleaner(args.cleaners)
    exp = Preprocessors('data/').select_experiment(args.experiment)
    onehot = True if args.encoding == 'onehot' else False
    mod_lib = oh if onehot else emb

    # Load tokenizers
    if args.tokenizer == 'spacy':
        selected_tok  = c.tokenize
    elif args.tokenizer == 'bpe':
        selected_tok = c.bpe_tokenize
    elif args.tokenizer == 'ekphrasis' and args.experiment == 'word':
        selected_tok = c.ekphrasis_tokenize
        annotate = {'elongated', 'emphasis'}
        flters = [f"<{filtr}>" for filtr in annotate]
        c._load_ekphrasis(annotate, flters)
    elif args.tokenizer == 'ekphrasis' and args.experiment == 'liwc':
        ekphr = c.ekphrasis_tokenize
        annotate = {'elongated', 'emphasis'}
        flters = [f"<{filtr}>" for filtr in annotate]
        c._load_ekphrasis(annotate, flters)

        def liwc_toks(doc):
            tokens = ekphr(doc)
            tokens = exp(tokens)
            return tokens
        selected_tok = liwc_toks
    tokenizer = selected_tok

    # Set up fields
    text = Field(tokenize = tokenizer, lower = True, batch_first = True)
    label = LabelField()
    fields = {'text': ('text', text), 'label': ('label', label)}  # Because we load from json we just need this.

    # Load training data
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
    elif args.main == 'waseem':
        train, dev, test = TabularDataset.splits(args.datadir, train = 'waseem_train.json',
                                                 validation = 'waseem_dev.json',
                                                 test = 'waseem_test.json',
                                                 format = 'json', skip_header = True, fields = fields)
    text.build_vocab(train)
    label.build_vocab(train)
    main = {'train': train, 'dev': dev, 'test': test, 'text': text, 'labels': label, 'name': args.main}

    # Dump Vocabulary files
    with open(f'{args.results}/vocabs/{args.main}_{args.encoding}_{args.experiment}.vocab', 'w',
              encoding = 'utf-8') as vocab_file:
        vocab_file.write(json.dumps(text.vocab.stoi))

    with open(f'{args.results}/vocabs/{args.main}_{args.encoding}_{args.experiment}.label', 'w',
              encoding = 'utf-8') as label_file:
        label_file.write(json.dumps(label.vocab.stoi))

    # Hyper parameters
    dropout = config.dropout
    nonlinearity = config.nonlinearity
    learning_rate = config.learning_rate
    epochs = config.epochs
    batch_size = config.batch_size

    # Set up training parameters
    if args.model in ['mlp', 'rnn']:
        params = {'hidden_dim': config.hidden, 'dropout': dropout, 'nonlinearity': nonlinearity}
        if not onehot:
            params.update({'embedding_dim': config.embedding})

        if args.model == 'mlp':
            mdl = mod_lib.MLPClassifier
        elif args.model == 'rnn':
            mdl = mod_lib.RNNClassifier
    elif args.model == 'lstm':
        params = dict(embedding_dim = config.embedding,
                      hidden_dim = config.hidden,
                      num_layers = args.num_layers,
                      dropout = dropout,
                      )
        mdl = mod_lib.LSTMClassifier
    elif args.model == 'cnn':
        params = {'window_sizes': [int(win) for win in config.window_sizes.split(',')], 'num_filters': config.filters,
                  'nonlinearity': nonlinearity}
        if onehot:
            params.update({'hidden_dim': config.hidden})
        else:
            params.update({'embedding_dim': config.embedding})
        mdl = mod_lib.CNNClassifier

    model_params = {'input_dim': len(text.vocab.stoi), 'output_dim': len(label.vocab.stoi), 'batch_first': True}
    model_params.update(params)

    # Initialize model
    model = mdl(**model_params)

    # Info about losses: https://bit.ly/3irxvYK
    if args.loss == 'nlll':
        loss = torch.nn.NLLLoss()
    elif args.loss == 'crossentropy':
        loss = torch.nn.CrossEntropyLoss()

    # Set optimizer and loss
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), learning_rate)
    elif args.optimizer == 'asgd':
        optimizer = torch.optim.ASGD(model.parameters(), learning_rate)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), learning_rate)

    # Batch train, dev and test set
    if not onehot:
        train_buckets = BucketIterator(dataset = main['train'], batch_size = batch_size,
                                       sort_key = lambda x: len(x))
        train = TorchtextExtractor('text', 'label', main['name'], train_buckets)
        dev_buckets = BucketIterator(dataset = main['dev'], batch_size = 64, sort_key = lambda x: len(x))
        dev = TorchtextExtractor('text', 'label', main['name'], dev_buckets)
        test_buckets = BucketIterator(dataset = main['test'], batch_size = 64, sort_key = lambda x: len(x))
        test = TorchtextExtractor('text', 'label', main['name'], test_buckets)
    else:
        train_buckets = BucketIterator(dataset = main['train'], batch_size = batch_size,
                                       sort_key = lambda x: len(x))
        train = TorchtextExtractor('text', 'label', main['name'], train_buckets, len(main['text'].vocab.stoi))
        dev_buckets = BucketIterator(dataset = main['dev'], batch_size = 64, sort_key = lambda x: len(x))
        dev = TorchtextExtractor('text', 'label', main['name'], dev_buckets, len(main['text'].vocab.stoi))
        test_buckets = BucketIterator(dataset = main['test'], batch_size = 64, sort_key = lambda x: len(x))
        test = TorchtextExtractor('text', 'label', main['name'], test_buckets)

    # Open output files
    base_path = f'{args.results}/{args.encoding}_{args.experiment}'
    enc = 'a' if os.path.isfile(f'{base_path}_train.tsv') else 'w'
    pred_enc = 'a' if os.path.isfile(f'{base_path}_preds.tsv') else 'w'

    train_writer = csv.writer(open(f"{base_path}_train.tsv", enc, encoding = 'utf-8'), delimiter = '\t')
    test_writer = csv.writer(open(f"{base_path}_test.tsv", enc, encoding = 'utf-8'), delimiter = '\t')
    pred_writer = csv.writer(open(f"{base_path}_preds.tsv", pred_enc, encoding = 'utf-8'), delimiter = '\t')

    hyper_info = [batch_size, epochs, learning_rate]
    model_hdr = ['Model', 'Input dim', 'Embedding dim', 'Hidden dim', 'Output dim', 'Window Sizes', '# Filters',
                 '# Layers', 'Dropout', 'Activation']
    if enc == 'w':
        metric_hdr = args.metrics + ['loss']
        hdr = ['Timestamp', 'Trained on', 'Evaluated on', 'Batch size', '# Epochs', 'Learning Rate'] + model_hdr
        hdr += metric_hdr
        test_writer.writerow(hdr)  # Don't include dev columns when writing test

        hdr += [f"dev {m}" for m in args.metrics] + ['dev loss']
        train_writer.writerow(hdr)

    pred_metric_hdr = args.metrics + ['loss']
    if pred_enc == 'w':
        hdr = ['Timestamp', 'Trained on', 'Evaluated on', 'Batch size', '# Epochs', 'Learning Rate'] + model_hdr
        hdr += ['Label', 'Prediction']
        pred_writer.writerow(hdr)

    gpu = True if args.gpu != -1 else False
    train_dict = dict(train = True,

                      # Set args
                      save_path = f"{args.save_model}{args.experiment}_{args.main}_best",
                      hyperopt = True,
                      gpu = gpu,
                      shuffle = False,

                      # Hyper-parameters
                      clip = 1.0,
                      epochs = epochs,
                      early_stopping = args.patience,
                      low = True if args.stop_metric == 'loss' else False,

                      # Model definitions
                      model = model,
                      loss = loss,
                      optimizer = optimizer,

                      # Dataset
                      batchers = train,
                      metrics = Metrics(args.metrics, args.display, args.stop_metric),
                      dev = dev,
                      dev_metrics = Metrics(args.metrics, args.display, args.stop_metric),

                      # Writing
                      writer = train_writer,
                      main_name = main['name'],
                      data_name = main['name'],
                      model_hdr = model_hdr,
                      metric_hdr = args.metrics + ['loss'],
                      hyper_info = hyper_info
                      )
    run_singletask_model(**train_dict)

    m_text = base.Field('text', train = True, label = False, ignore = False, ix = 1, cname = 'text')
    m_label = base.Field('label', train = False, label = True, cname = 'label', ignore = False, ix = 2)
    for aux in tqdm(args.aux, desc = "Loading test sets"):
        if aux == 'waseem':
            loaded = GeneralDataset(args.datadir, 'json', [m_text, m_label], 'waseem', 'waseem_binary_train.json',
                                    None, 'waseem_binary_test.json', None, None, None, None, tokenizer, None, None,
                                    None, None, None, True)

            loaded.stoi = text.vocab.stoi
            loaded.ltoi = label.vocab.stoi
            loaded.process_labels(loaded.test)
            batcher = process_and_batch(loaded, loaded.test, 64, onehot)
        elif aux == 'waseem_hovy':
            loaded = GeneralDataset(args.datadir, 'json', [m_text, m_label], 'waseem-hovy',
                                    'waseem_hovy_binary_train.json', None, 'waseem_hovy_binary_test.json', None, None,
                                    None, None, tokenizer, None, None, None, None, None, True)
            loaded.stoi = text.vocab.stoi
            loaded.ltoi = label.vocab.stoi
            loaded.process_labels(loaded.test)
            batcher = process_and_batch(loaded, loaded.test, 64, onehot)
        elif aux == 'wulczyn':
            loaded = GeneralDataset(args.datadir, 'json', [m_text, m_label], 'wulczyn', 'wulczyn_binary_train.json',
                                    None, 'wulczyn_binary_test.json', None, None, None, None, tokenizer, None, None,
                                    None, None, None, True)
            loaded.stoi = text.vocab.stoi
            loaded.ltoi = label.vocab.stoi
            loaded.process_labels(loaded.test)
            batcher = process_and_batch(loaded, loaded.test, 64, onehot)
        elif aux == 'davidson':
            loaded = GeneralDataset(args.datadir, 'json', [m_text, m_label], 'davidson', 'davidson_binary_train.json',
                                    None, 'davidson_binary_test.json', None, None, None, None, tokenizer, None, None,
                                    None, None, None, True)
            loaded.stoi = text.vocab.stoi
            loaded.ltoi = label.vocab.stoi
            loaded.process_labels(loaded.test)
            batcher = process_and_batch(loaded, loaded.test, 64, onehot)
        elif aux == 'garcia':
            loaded = GeneralDataset(args.datadir, 'json', [m_text, m_label], 'garcia', 'garcia_binary_train.json',
                                    None, 'garcia_binary_test.json', None, None, None, None, tokenizer, None, None,
                                    None, None, None, True)
            loaded.stoi = text.vocab.stoi
            loaded.ltoi = label.vocab.stoi
            loaded.process_labels(loaded.test)
            batcher = process_and_batch(loaded, loaded.test, 64, onehot)

        eval_dict = dict(train = False,

                         # Evaluating
                         model = model,
                         batchers = batcher,
                         loss = loss,
                         metrics = Metrics(args.metrics, args.display, args.stop_metric),
                         gpu = gpu,

                         # Storing predictions
                         store = False,

                         # Prediction writer
                         writer = pred_writer,
                         model_hdr = model_hdr,
                         data_name = aux,
                         main_name = args.main,
                         hyper_info = hyper_info,
                         data = batcher,
                         dataset = loaded,
                         train_field = 'text',
                         label_field = 'label',
                         )
        run_singletask_model(**eval_dict)
