import os
import csv
import torch
import optuna
import numpy as np
from tqdm import tqdm
import mlearn.modeling.onehot as oh
import mlearn.modeling.linear as lin
import mlearn.data.loaders as loaders
import mlearn.modeling.embedding as emb
from mlearn.utils.metrics import Metrics
from mlearn.data.clean import Cleaner, Preprocessors
from jsonargparse import ArgumentParser, ActionConfigFile
from mlearn.utils.train import run_singletask_model as run_model
from mlearn.utils.pipeline import process_and_batch, param_selection


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
    optimisable = param_selection(trial, params)

    # TODO Think of a way to not hardcode this.
    training.update(dict(
        batchers = process_and_batch(dataset, dataset.data, optimisable['batch_size'], modeling['onehot']),
        hidden_dim = optimisable['hidden'] if 'hidden' in optimisable else None,
        embedding_dim = optimisable['embedding'] if 'embedding' in optimisable else None,
        hyper_info = [optimisable['batch_size'], optimisable['epochs'], optimisable['learning_rate']],
        dropout = optimisable['dropout'],
        nonlinearity = optimisable['nonlinearity'],
        epochs = optimisable['epochs'],
        hyperopt = trial
    ))
    breakpoint()
    training['model'] = model(**training)
    training.update(dict(
        loss = modeling['loss'](),
        optimizer = modeling['optimizer'](training['model'].parameters(), optimisable['learning_rate']),
        metrics = Metrics(modeling['metrics'], modeling['display'], modeling['stop']),
        dev_metrics = Metrics(modeling['metrics'], modeling['display'], modeling['stop'])
    ))

    run_model(train = True, writer = modeling['train_writer'], **training)

    if direction == 'minimize':
        metric = training['dev_metrics'].loss
    else:
        metric = np.mean(training['dev_metrics'].scores[modeling['display']])

    eval = dict(
        model = training['model'],
        loss = training['loss'],
        metrics = Metrics(modeling['metrics'], modeling['display'], modeling['stop']),
        gpu = training['gpu'],
        data = modeling['main'].test,
        dataset = modeling['main'],
        hyper_info = training['hyper_info'],
        model_hdr = training['model_hdr'],
        metric_hdr = training['metric_hdr'],
        main_name = training['main_name'],
        data_name = training['main_name'],
        train_field = 'text',
        label_field = 'label',
        store = False,
    )

    for dataset, batcher in zip(modeling['test_sets'], modeling['test_batcher']):
        eval['batcher'] = batcher
        eval['data'] = dataset.test
        eval['data_name'] = dataset.name
        run_model(train = False, writer = modeling['test_writer'], pred_writer = None, **eval)

    return metric


if __name__ == "__main__":
    parser = ArgumentParser(description = "Run Experiments to generalise models.")

    # For all models
    parser.add_argument("--train", help = "Choose train data: Davidson, Waseem, Waseem and Hovy, Wulczyn, and Garcia.",
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

    # Model architecture
    parser.add_argument("--embedding", help = "Set the embedding dimension.", default = [300], type = int, nargs = '+')
    parser.add_argument("--hidden", help = "Set the hidden dimension.", default = [128], type = int, nargs = '+')
    parser.add_argument("--layers", help = "Set the number of layers.", default = 1, type = int)
    parser.add_argument("--window_sizes", help = "Set the window sizes.", nargs = '+', default = [[2, 3, 4]],
                        type = list)
    parser.add_argument("--filters", help = "Set the number of filters for CNN.", default = [128], type = int,
                        nargs = '+')
    # parser.add_argument("--max_feats", help = "Set the number of features for CNN.", default = 100, type = int)
    parser.add_argument("--optimizer", help = "Optimizer to use.", default = 'adam', type = str.lower)
    parser.add_argument("--loss", help = "Loss to use.", default = 'nlll', type = str.lower)
    parser.add_argument('--encoding', help = "Select encoding to be used: Onehot, Embedding, Tfidf, Count",
                        default = 'embedding', type = str.lower)
    parser.add_argument('--tokenizer', help = "select the tokenizer to be used: Spacy, BPE", default = 'spacy',
                        type = str.lower)

    # Model (hyper) parameters
    parser.add_argument("--epochs", help = "Set the number of epochs.", default = [200], type = int, nargs = '+')
    parser.add_argument("--batch_size", help = "Set the batch size.", default = [64], type = int, nargs = '+')
    parser.add_argument("--dropout", help = "Set value for dropout.", default = [0.0], type = float, nargs = '+')
    parser.add_argument('--learning_rate', help = "Set the learning rate for the model.", default = [0.01],
                        type = float, nargs = '+')
    parser.add_argument("--nonlinearity", help = "Set activation function for neural nets.", default = ['tanh'],
                        type = str.lower, nargs = '+')
    parser.add_argument("--hyperparams", help = "List of names of the hyper parameters to be searched.",
                        default = ['epochs'], type = str.lower, nargs = '+')
    parser.add_argument("--n_trials", help = "Set the number of hyper-parameter search trials to run.", default = 10,
                        type = int)

    # Experiment parameters
    parser.add_argument('--shuffle', help = "Shuffle dataset between epochs", type = bool, default = True)
    parser.add_argument('--gpu', help = "Set to run on GPU", type = bool, default = False)
    parser.add_argument('--seed', help = "Set the random seed.", type = int, default = 32)
    parser.add_argument("--experiment", help = "Set experiment to run.", default = "word", type = str.lower)
    parser.add_argument("--slur_window", help = "Set window size for slur replacement.", default = None, type = int,
                        nargs = '+')
    parser.add_argument('--cfg', action = ActionConfigFile, default = None)

    args = parser.parse_args()

    # Set seeds
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(1)

    if 'f1' in args.metrics + [args.display, args.stop_metric]:
        for i, m in enumerate(args.metrics):
            if 'f1' in m:
                args.metrics[i] = 'f1-score'
        if args.display == 'f1':
            args.display = 'f1-score'
        if args.stop_metric == 'f1':
            args.display = 'f1-score'

    if args.encoding == 'embedding':
        mod_lib = emb
        onehot = False
    elif args.encoding == 'onehot':
        mod_lib = oh
        onehot = True
    elif args.encoding in ['tfidf', 'count']:
        mod_lib = lin

    c = Cleaner(args.cleaners)
    p = Preprocessors(args.datadir)
    tokenizer = c.tokenize if args.tokenizer == 'spacy' else c.bpe_tokenize
    experiment = p.select_experiment(args.experiment, args.slur_window)

    if args.train == 'davidson':
        main = loaders.davidson(tokenizer, args.datadir, preprocessor = experiment,
                                label_processor = loaders.davidson_to_binary, stratify = 'label', skip_header = True)
        test_sets = [main,
                     loaders.wulczyn(tokenizer, args.datadir, preprocessor = experiment, stratify = 'label',
                                     skip_header = True),
                     loaders.garcia(tokenizer, args.datadir, preprocessor = experiment,
                                    label_processor = loaders.binarize_garcia, stratify = 'label', skip_header = True),
                     loaders.waseem(tokenizer, args.datadir, preprocessor = experiment,
                                    label_processor = loaders.waseem_to_binary, stratify = 'label'),
                     loaders.waseem_hovy(tokenizer, args.datadir, preprocessor = experiment,
                                         label_processor = loaders.waseem_to_binary,
                                         stratify = 'label')
                     ]

    elif args.train == 'waseem':
        main = loaders.waseem(tokenizer, args.datadir, preprocessor = experiment,
                              label_processor = loaders.waseem_to_binary, stratify = 'label'),
        test_sets = [main,
                     loaders.wulczyn(tokenizer, args.datadir, preprocessor = experiment, stratify = 'label',
                                     skip_header = True),
                     loaders.garcia(tokenizer, args.datadir, preprocessor = experiment,
                                    label_processor = loaders.binarize_garcia, stratify = 'label', skip_header = True),
                     loaders.davidson(tokenizer, args.datadir, preprocessor = experiment,
                                      label_processor = loaders.davidson_to_binary, stratify = 'label',
                                      skip_header = True),
                     loaders.waseem_hovy(tokenizer, args.datadir, preprocessor = experiment,
                                         label_processor = loaders.waseem_to_binary,
                                         stratify = 'label')
                     ]

    elif args.train == 'waseem_hovy':
        main = loaders.waseem_hovy(tokenizer, args.datadir, preprocessor = experiment,
                                   label_processor = loaders.waseem_to_binary,
                                   stratify = 'label')
        test_sets = [main,
                     loaders.davidson(tokenizer, args.datadir, preprocessor = experiment,
                                      label_processor = loaders.davidson_to_binary, stratify = 'label',
                                      skip_header = True),
                     loaders.wulczyn(tokenizer, args.datadir, preprocessor = experiment, stratify = 'label',
                                     skip_header = True),
                     loaders.garcia(tokenizer, args.datadir, preprocessor = experiment,
                                    label_processor = loaders.binarize_garcia,
                                    stratify = 'label', skip_header = True),
                     loaders.waseem(tokenizer, args.datadir, preprocessor = experiment,
                                    label_processor = loaders.waseem_to_binary,
                                    stratify = 'label')
                     ]

    elif args.train == 'garcia':
        main = loaders.garcia(tokenizer, args.datadir, preprocessor = experiment,
                              label_processor = loaders.binarize_garcia, stratify = 'label', skip_header = True),
        test_sets = [main,
                     loaders.davidson(tokenizer, args.datadir, preprocessor = experiment,
                                      label_processor = loaders.davidson_to_binary, stratify = 'label',
                                      skip_header = True),
                     loaders.wulczyn(tokenizer, args.datadir, preprocessor = experiment, stratify = 'label',
                                     skip_header = True),
                     loaders.waseem(tokenizer, args.datadir, preprocessor = experiment,
                                    label_processor = loaders.waseem_to_binary,
                                    stratify = 'label'),
                     loaders.waseem_hovy(tokenizer, args.datadir, preprocessor = experiment,
                                         label_processor = loaders.waseem_to_binary,
                                         stratify = 'label')
                     ]

    elif args.train == 'wulczyn':
        main = loaders.wulczyn(tokenizer, args.datadir, preprocessor = experiment, stratify = 'label',
                               skip_header = True)
        test_sets = [main,
                     loaders.davidson(tokenizer, args.datadir, preprocessor = experiment,
                                      label_processor = loaders.davidson_to_binary, stratify = 'label',
                                      skip_header = True),
                     loaders.garcia(tokenizer, args.datadir, preprocessor = experiment,
                                    label_processor = loaders.binarize_garcia,
                                    stratify = 'label', skip_header = True),
                     loaders.waseem(tokenizer, args.datadir, preprocessor = experiment,
                                    label_processor = loaders.waseem_to_binary,
                                    stratify = 'label'),
                     loaders.waseem_hovy(tokenizer, args.datadir, preprocessor = experiment,
                                         label_processor = loaders.waseem_to_binary,
                                         stratify = 'label')
                     ]

    # Build token and label vocabularies
    main.build_token_vocab(main.data)
    main.build_label_vocab(main.data)

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

    train_args = dict(
        # For writers
        model_hdr = model_hdr,
        metric_hdr = args.metrics + ['loss'],

        # Batch dev
        dev = process_and_batch(main, main.dev, 64, onehot),

        # Set model dimensionality
        batch_first = True,
        early_stopping = args.patience,
        num_layers = args.layers,
        window_sizes = args.window_sizes[0],
        num_filters = args.filters[0],
        # max_feats = args.max_feats,
        input_dim = main.vocab_size(),
        output_dim = main.label_count(),

        # Main task information
        main_name = main.name,

        # Met information
        shuffle = args.shuffle,
        gpu = args.gpu,
        save_path = f"{args.save_model}{args.experiment}_best",
        low = True if args.stop_metric == 'loss' else False,
    )

    # Set models to iterate over
    models = []
    for m in args.model:
        if m == 'mlp':
            models.append(mod_lib.MLPClassifier)
        if m == 'lstm':
            models.append(mod_lib.LSTMClassifier)
        if m == 'cnn':
            models.append(mod_lib.CNNClassifier)
        if m == 'rnn':
            models.append(mod_lib.RNNClassifier)
        if m == 'all':
            models = [mod_lib.MLPClassifier,
                      mod_lib.CNNClassifier,
                      mod_lib.LSTMClassifier,
                      mod_lib.RNNClassifier]

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
        test_batcher = [process_and_batch(main, data.test, 64, onehot) for data in test_sets],
        main = main,
        train_writer = train_writer,
        test_writer = test_writer,
        pred_writer = None,
        test_sets = test_sets,
        onehot = onehot
    )

    with tqdm(models, desc = "Model Iterator") as m_loop:
        params = {param: getattr(args, param) for param in args.hyperparams}  # Get hyper-parameters to search
        breakpoint()
        direction = 'minimize' if args.display == 'loss' else 'maximize'
        study = optuna.create_study(study_name = 'Vocab Redux', direction = direction)
        trial_file = open(f"{base}.trials", 'a', encoding = 'utf-8')

        for m in models:
            study.optimize(lambda trial: sweeper(trial, train_args, main, params, m, modeling, direction),
                           n_trials = 100, gc_after_trial = True, n_jobs = 1, show_progress_bar = True)

            print(f"Model: {m}", file = trial_file)
            print(f"Best parameters: {study.best_params}", file = trial_file)
            print(f"Best trial: {study.best_trial}", file = trial_file)
            print(f"All trials: {study.trials}", file = trial_file)
