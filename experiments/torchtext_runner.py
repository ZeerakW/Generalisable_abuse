import os
import csv
import json
import torch
import optuna
import numpy as np
from tqdm import tqdm
from mlearn import base
from mlearn.utils.metrics import Metrics
from mlearn.modeling import onehot as oh
from mlearn.modeling import embedding as emb
from mlearn.data.dataset import GeneralDataset
from mlearn.utils.pipeline import param_selection
from mlearn.utils.evaluate import eval_torch_model
from mlearn.data.batching import TorchtextExtractor
from mlearn.data.clean import Cleaner, Preprocessors
from mlearn.utils.train import train_singletask_model
from jsonargparse import ArgumentParser, ActionConfigFile
from torchtext.data import TabularDataset, Field, LabelField, BucketIterator


def sweeper(trial, training: dict, dataset: list, params: dict, model, modeling: dict, direction: str):
    """
    The function that contains all loading and setting of values and running the sweeps.

    :trial: The Optuna trial.
    :training (dict): Dictionary containing training modeling.
    :datasets (list): List of datasets objects.
    :params (dict): A dictionary of the different tunable parameters and their values.
    :model: The model to train.
    :modeling (dict): The arguments for the model and metrics objects.
    """
    model_params = ['batch_size', 'epochs', 'learning_rate', 'dropout', 'embedding']
    if train_args['model_name'] in ['rnn', 'mlp']:
        model_params += ['hidden', 'nonlinearity']
    if train_args['model_name'] == 'lstm':
        model_params += ['hidden']
    if train_args['model_name'] == 'cnn':
        model_params += ['window_sizes', 'filters', 'nonlinearity']
    param_space = {p: params[p] for p in set(model_params)}

    optimisable = param_selection(trial, param_space)
    optimisable['batch_size'] = 64
    optimisable['epochs'] = 200
    optimisable['learning_rate'] = 0.01
    optimisable['dropout'] = 0.1
    optimisable['embedding'] = 200
    optimisable['hidden'] = 200
    optimisable['nonlinearity'] = 'relu'
    optimisable['filters'] = 128
    optimisable['window_sizes'] = [2, 3, 4]

    if not modeling['onehot']:
        train_buckets = BucketIterator(dataset = dataset['train'], batch_size = optimisable['batch_size'],
                                       sort_key = lambda x: len(x), shuffle = training['shuffle'])
        train = TorchtextExtractor('text', 'label', dataset['name'], train_buckets)
    else:
        train_buckets = BucketIterator(dataset = dataset['train'], batch_size = optimisable['batch_size'],
                                       sort_key = lambda x: len(x), shuffle = training['shuffle'])
        train = TorchtextExtractor('text', 'label', dataset['name'], train_buckets, len(main['text'].vocab.stoi))
    training['shuffle'] = False

    # TODO Think of a way to not hardcode this.
    training.update(dict(
        batchers = train,
        hidden_dim = optimisable['hidden'] if 'hidden' in optimisable else None,
        embedding_dim = optimisable['embedding'] if 'embedding' in optimisable else None,
        hyper_info = [optimisable['batch_size'], optimisable['epochs'], optimisable['learning_rate']],
        dropout = optimisable['dropout'],
        nonlinearity = optimisable['nonlinearity'],
        epochs = optimisable['epochs'],
        hyperopt = trial,
        data_name = dataset['name'],
        metrics = Metrics(modeling['metrics'], modeling['display'], modeling['stop']),
        dev_metrics = Metrics(modeling['metrics'], modeling['display'], modeling['stop']),
        clip = 1.0,
    ))
    training['model'] = model(**training)
    training['loss'] = modeling['loss']()
    training['optimizer'] = modeling['optimizer'](training['model'].parameters(), optimisable['learning_rate'])

    train_singletask_model(train = True, writer = modeling['train_writer'], **training)

    eval = dict(
        model = training['model'],
        loss = training['loss'],
        metrics = Metrics(modeling['metrics'], modeling['display'], modeling['stop']),
        gpu = training['gpu'],
        data = training['test'],
        dataset = modeling['main'],
        hyper_info = training['hyper_info'],
        model_hdr = training['model_hdr'],
        metric_hdr = training['metric_hdr'],
        main_name = training['main_name'],
        data_name = training['main_name'],
        train_field = 'text',
        label_field = 'label',
        store = False,
        batchers = training['test'],
    )

    eval_torch_model(**eval)

    if direction == 'minimize':
        return training['dev_metrics'].loss
    else:
        return np.mean(training['dev_metrics'].scores[modeling['display']])


    # TODO: Need to do some processing of all of the test sets.
    # TODO Evalaute on test set
    # TODO Evaluate on all other test sets.


# batched_test = []
# for dataset in test_sets:  # TODO From here
#     ds = [indices(doc) for doc in dataset]
#     if not onehot:
#         test = TorchtextExtractor('text', 'label', main['name'], test_buckets)
#     else:
#         test = TorchtextExtractor('text', 'label', main['name'], test_buckets, len(main['text'].vocab.stoi))
#     dataset['test_buckets'] = test_buckets
#     batched_test.append(test)


if __name__ == "__main__":
    parser = ArgumentParser(description = "Run Experiments using MTL.")

    # For all modesl
    parser.add_argument("--main", help = "Choose train data: Davidson, Waseem, Waseem and Hovy, Wulczyn, and Garcia.",
                        type = str.lower, default = 'Davidson')
    parser.add_argument("--model", help = "Choose the model to be run: CNN, RNN, LSTM, MLP, LR.", nargs = '+',
                        default = ['mlp'], type = str.lower)
    parser.add_argument("--save_model", help = "Directory to store models in.", default = 'results/models/')
    parser.add_argument("--results", help = "Set file to output results to.", default = 'results/')
    parser.add_argument("--datadir", help = "Path to the datasets.", default = 'data/')
    parser.add_argument("--cleaners", help = "Set the cleaning routines to be used.", nargs = '+', default = None)
    parser.add_argument("--metrics", help = "Set the metrics to be used.", nargs = '+', default = ["f1"],
                        type = str.lower)
    parser.add_argument("--stop_metric", help = "Set the metric to be used for early stopping", default = "loss")
    parser.add_argument("--display", help = "Metric to display in TQDM loops.", default = 'f1-score')
    parser.add_argument("--patience", help = "Set the number of epochs to keep trying to find a new best",
                        default = None, type = int)
    parser.add_argument("--aux", help = "Specify the auxiliary datasets to be loaded.", type = str, nargs = '+')
    parser.add_argument("--n_trials", help = "Set number of trials to run.", type = int)

    # Model architecture
    parser.add_argument("--window_sizes", help = "Set the window sizes for CNN.", default = [(2, 3, 4)], type = tuple,
                        nargs = '+')
    parser.add_argument("--filters", help = "Set the number of filters for CNN.", default = [128], type = int,
                        nargs = '+')
    parser.add_argument("--embedding", help = "Set the embedding dimension.", default = 100, type = int,
                        nargs = '+')
    parser.add_argument("--hidden", help = "Set the hidden dimension.", default = 128, type = int,
                        nargs = '+')
    parser.add_argument("--shared", help = "Set the shared dimension", default = [256], type = int, nargs = '+')
    parser.add_argument("--optimizer", help = "Optimizer to use.", default = 'adam', type = str.lower)
    parser.add_argument("--loss", help = "Loss to use.", default = 'nlll', type = str.lower)
    parser.add_argument('--encoding', help = "Select encoding to be used: Onehot, Index, Tfidf, Count",
                        default = 'index', type = str.lower)
    parser.add_argument('--tokenizer', help = "select the tokenizer to be used: Spacy, Ekphrasis, BPE",
                        default = 'ekphrasis', type = str.lower)
    parser.add_argument("--layers", help = "Set the number of layers.", default = 1, type = int)

    # Model (hyper) parameters
    parser.add_argument("--epochs", help = "Set the number of epochs.", default = [200], type = int, nargs = '+')
    parser.add_argument("--batch_size", help = "Set the batch size.", default = [64], type = int, nargs = '+')
    # parser.add_argument("--dropout", help = "Set value for dropout.", default = [0.0, 0.0], type = float, nargs = '+')
    parser.add_argument("--dropout.high", help = "Set upper limit for dropout.", default = 1.0, type = float)
    parser.add_argument("--dropout.low", help = "Set lower limit for dropout.", default = 0.0, type = float)
    # parser.add_argument('--learning_rate', help = "Set the learning rate for the model.", default = [0.01],
    #                     type = float, nargs = '+')
    parser.add_argument('--learning_rate.high', help = "Set the upper limit for the learning rate.", default = [1.0],
                        type = float)
    parser.add_argument('--learning_rate.low', help = "Set the lower limit for the learning rate.", default = [0.0001],
                        type = float)

    parser.add_argument("--nonlinearity", help = "Set nonlinearity function for neural nets.", default = ['tanh'],
                        type = str.lower, nargs = '+')
    parser.add_argument("--hyperparams", help = "List of names of the hyper parameters to be searched.",
                        default = ['epochs'], type = str.lower, nargs = '+')

    # Experiment parameters
    parser.add_argument("--batches_epoch", help = "Set the number of batches per epoch", type = int, default = None)
    parser.add_argument("--loss_weights", help = "Set the weight of each task", type = int, default = None,
                        nargs = '+')
    parser.add_argument('--shuffle', help = "Shuffle dataset between epochs", type = bool, default = True)
    parser.add_argument('--gpu', help = "Set to run on GPU", type = int, default = 0)
    parser.add_argument('--seed', help = "Set the random seed.", type = int, default = 32)
    parser.add_argument("--experiment", help = "Set experiment to run.", default = "word", type = str.lower)
    parser.add_argument('--cfg', action = ActionConfigFile, default = None)
    args = parser.parse_args()

    if 'f1' in args.metrics:
        args.metrics[args.metrics.index('f1')] = 'f1-score'
    if args.display == 'f1':
        args.display = 'f1-score'
    if args.stop_metric == 'f1':
        args.stop_metric = 'f1-score'

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize experiment
    train_dict = dict(datadir = args.datadir,
                      gpu = args.gpu,
                      hyperopt = True,
                      save_path = args.save_model
                      )

    c = Cleaner(args.cleaners)
    experiment = Preprocessors('data/').select_experiment(args.experiment)
    onehot = True if args.encoding == 'onehot' else False
    filters = None

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
        filters = [f"<{filtr}>" for filtr in annotate]
        c._load_ekphrasis(annotate, filters)

        def liwc_toks(doc):
            tokens = ekphr(doc)
            tokens = args.experimenteriment(tokens)
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
                                                 format = 'json', skip_header = False, fields = fields)
    elif args.main == 'wulczyn':
        train, dev, test = TabularDataset.splits(args.datadir, train = 'wulczyn_train.json',
                                                 validation = 'wulczyn_dev.json',
                                                 test = 'wulczyn_test.json',
                                                 format = 'json', skip_header = False, fields = fields)
    elif args.main == 'wasem':
        train, dev, test = TabularDataset.splits(args.datadir, train = 'waseem_train.json',
                                                 validation = 'waseem_dev.json',
                                                 test = 'waseem_test.json',
                                                 format = 'json', skip_header = False, fields = fields)
    text.build_vocab(train)
    label.build_vocab(train)
    main = {'train': train, 'dev': dev, 'test': test, 'text': text, 'labels': label, 'name': args.main}

    with open(f'{args.results}/vocabs/{args.main}_{args.encoding}_{args.experiment}.vocab', 'w',
              encoding = 'utf-8') as vocab_file:
        vocab_file.write(json.dumps(text.vocab.stoi))

    with open(f'{args.results}/vocabs/{args.main}_{args.encoding}_{args.experiment}.label', 'w',
             encoding = 'utf-8') as label_file:
        label_file.write(json.dumps(label.vocab.stoi))

    auxiliary = []
    m_text = base.Field('text', train = True, label = False, ignore = False, ix = 1, cname = 'text')
    m_label = base.Field('label', train = False, label = True, cname = 'label', ignore = False, ix = 2)
    for aux in args.aux:
        if aux == 'waseem':
            loaded = GeneralDataset(args.datadir, 'json', [m_text, m_label], 'waseem', 'waseem_binary_train.json',
                                    None, 'waseem_binary_test.json', None, None, None, None, tokenizer, None, None,
                                    None, None, None, True)
        elif aux == 'waseem_hovy':
            loaded = GeneralDataset(args.datadir, 'json', [m_text, m_label], 'waseem-hovy',
                                    'waseem_hovy_binary_train.json', None, 'waseem_hovy_binary_test.json', None, None,
                                    None, None, tokenizer, None, None, None, None, None, True)
        elif aux == 'wulczyn':
            loaded = GeneralDataset(args.datadir, 'json', [m_text, m_label], 'wulczyn', 'wulczyn_binary_train.json',
                                    None, 'wulczyn_binary_test.json', None, None, None, None, tokenizer, None, None,
                                    None, None, None, True)
        elif aux == 'davidson':
            loaded = GeneralDataset(args.datadir, 'json', [m_text, m_label], 'davidson', 'davidson_binary_train.json',
                                    None, 'davidson_binary_test.json', None, None, None, None, tokenizer, None, None,
                                    None, None, None, True)
        elif aux == 'garcia':
            loaded = GeneralDataset(args.datadir, 'json', [m_text, m_label], 'garcia', 'garcia_binary_train.json',
                                    None, 'garcia_binary_test.json', None, None, None, None, tokenizer, None, None,
                                    None, None, None, True)
        auxiliary.append(loaded)

    # Open output files
    base = f'{args.results}/{args.encoding}_{args.experiment}'
    enc = 'a' if os.path.isfile(f'{base}_train.tsv') else 'w'
    pred_enc = 'a' if os.path.isfile(f'{base}_preds.tsv') else 'w'

    train_writer = csv.writer(open(f"{base}_train.tsv", enc, encoding = 'utf-8'), delimiter = '\t')
    test_writer = csv.writer(open(f"{base}_test.tsv", enc, encoding = 'utf-8'), delimiter = '\t')
    pred_writer = csv.writer(open(f"{base}_preds.tsv", pred_enc, encoding = 'utf-8'), delimiter = '\t')

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

    if not onehot:
        dev_buckets = BucketIterator(dataset = main['dev'], batch_size = 64, sort_key = lambda x: len(x))
        dev = TorchtextExtractor('text', 'label', main['name'], dev_buckets)
        test_buckets = BucketIterator(dataset = main['test'], batch_size = 64, sort_key = lambda x: len(x))
        test = TorchtextExtractor('text', 'label', main['name'], test_buckets)
    else:
        dev_buckets = BucketIterator(dataset = main['dev'], batch_size = 64, sort_key = lambda x: len(x))
        dev = TorchtextExtractor('text', 'label', main['name'], dev_buckets, len(main['text'].vocab.stoi))
        test_buckets = BucketIterator(dataset = main['test'], batch_size = 64, sort_key = lambda x: len(x))
        test = TorchtextExtractor('text', 'label', main['name'], test_buckets)

    train_args = dict(
        # For writers
        model_hdr = model_hdr,
        metric_hdr = args.metrics + ['loss'],

        # Batch dev
        dev = dev,
        test = test,

        # Set model dimensionality
        batch_first = True,
        early_stopping = args.patience,
        num_layers = args.layers,
        input_dim = len(main['text'].vocab.stoi),
        output_dim = len(main['labels'].vocab.stoi),

        # Main task information
        main_name = main['name'],

        # Met information
        shuffle = args.shuffle,
        gpu = args.gpu,
        save_path = f"{args.save_model}{args.experiment}_{args.main}_best",
        low = True if args.stop_metric == 'loss' else False,
    )
    train_args.update(train_dict)

    # Set optimizer and loss
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD
    elif args.optimizer == 'asgd':
        optimizer = torch.optim.ASGD
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW

    # Info about losses: https://bit.ly/3irxvYK
    if args.loss == 'nlll':
        loss = torch.nn.NLLLoss
    elif args.loss == 'crossentropy':
        loss = torch.nn.CrossEntropyLoss

    modeling = dict(
        optimizer = optimizer,
        loss = loss,
        metrics = args.metrics,
        display = args.display,
        stop = args.stop_metric,
        main = main,
        train_writer = train_writer,
        test_writer = test_writer,
        pred_writer = None,
        onehot = onehot,
        aux = auxiliary
    )

    mod_lib = oh if onehot == 'onehot' else emb
    models, model_names = [], []
    for m in args.model:
        if m == 'mlp':
            models.append(mod_lib.MLPClassifier)
            model_names.append(m)
        if m == 'lstm':
            models.append(mod_lib.LSTMClassifier)
            model_names.append(m)
        if m == 'cnn':
            models.append(mod_lib.CNNClassifier)
            model_names.append(m)
        if m == 'rnn':
            models.append(mod_lib.RNNClassifier)
            model_names.append(m)
        if m == 'all':
            models = [mod_lib.MLPClassifier,
                      mod_lib.CNNClassifier,
                      mod_lib.LSTMClassifier,
                      mod_lib.RNNClassifier]
            model_names.extend(['mlp', 'cnn', 'lstm', 'rnn'])

    with tqdm(models, desc = "Model Iterator") as m_loop:
        params = {param: getattr(args, param) for param in args.hyperparams}  # Get hyper-parameters to search
        direction = 'minimize' if args.display == 'loss' else 'maximize'
        trial_file = open(f"{base}_{args.main}.trials", 'a', encoding = 'utf-8')

        for i, m in enumerate(m_loop):
            train_args['model_name'] = model_names[i]
            study = optuna.create_study(study_name = 'Vocab Redux', direction = direction)
            study.optimize(lambda trial: sweeper(trial, train_args, main, params, m, modeling, direction),
                           n_trials = args.n_trials, gc_after_trial = True, n_jobs = 1, show_progress_bar = True)

            print(f"Model: {m}", file = trial_file)
            print(f"Best parameters: {study.best_params}", file = trial_file)
            print(f"Best trial: {study.best_trial}", file = trial_file)
            print(f"All trials: {study.trials}", file = trial_file)
