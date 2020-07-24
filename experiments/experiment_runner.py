import os
import csv
import torch
import argparse
import numpy as np
from tqdm import tqdm
import mlearn.modeling.onehot as oh
import mlearn.modeling.linear as lin
import mlearn.data.loaders as loaders
import mlearn.modeling.embedding as emb
from mlearn.utils.metrics import Metrics
from mlearn.utils.pipeline import process_and_batch
from mlearn.data.clean import Cleaner, Preprocessors
from mlearn.utils.train import run_singletask_model as run_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run Experiments to generalise models.")

    # For all models
    parser.add_argument("--train", help = "Choose train data: davidson, Waseem, Waseem and Hovy, wulczyn, and garcia.",
                        type = str.lower)
    parser.add_argument("--model", help = "Choose the model to be run: CNN, RNN, LSTM, MLP, LR.", nargs = '+',
                        default = ['mlp'], type = str.lower)
    parser.add_argument("--save_model", help = "Directory to store models in.", default = 'results/model/')
    parser.add_argument("--results", help = "Set file to output results to.", default = 'results/')
    parser.add_argument("--cleaners", help = "Set the cleaning routines to be used.", nargs = '+', default = None)
    parser.add_argument("--metrics", help = "Set the metrics to be used.", nargs = '+', default = ["f1"],
                        type = str.lower)
    parser.add_argument("--stop_metrics", help = "Set the metric to be used for early stopping", default = "loss")
    parser.add_argument("--patience", help = "Set the number of epochs to keep trying to find a new best",
                        default = None, type = int)
    parser.add_argument("--display", help = "Metric to display in TQDM loops.", default = 'accuracy')
    parser.add_argument("--datadir", help = "Path to the datasets.", default = 'data/')

    # Model architecture
    parser.add_argument("--embedding", help = "Set the embedding dimension.", default = [300], type = int, nargs = '+')
    parser.add_argument("--hidden", help = "Set the hidden dimension.", default = [128], type = int, nargs = '+')
    parser.add_argument("--layers", help = "Set the number of layers.", default = 1, type = int)
    parser.add_argument("--window_sizes", help = "Set the window sizes.", nargs = '+', default = [2, 3, 4], type = int)
    parser.add_argument("--filters", help = "Set the number of filters for CNN.", default = 128, type = int)
    # parser.add_argument("--max_feats", help = "Set the number of features for CNN.", default = 100, type = int)
    parser.add_argument("--optimizer", help = "Optimizer to use.", default = 'adam', type = str.lower)
    parser.add_argument("--loss", help = "Loss to use.", default = 'nlll', type = str.lower)
    parser.add_argument('--encoding', help = "Select encoding to be used: Onehot, Embedding, Tfidf, Count",
                        default = ['embedding'], type = str.lower)

    # Model (hyper) parameters
    parser.add_argument("--epochs", help = "Set the number of epochs.", default = [200], type = int, nargs = '+')
    parser.add_argument("--batch_size", help = "Set the batch size.", default = [64], type = int, nargs = '+')
    parser.add_argument("--dropout", help = "Set value for dropout.", default = [0.0], type = float, nargs = '+')
    parser.add_argument('--learning_rate', help = "Set the learning rate for the model.", default = [0.01],
                        type = float, nargs = '+')

    # Experiment parameters
    parser.add_argument('--shuffle', help = "Shuffle dataset between epochs", action = 'store_true', default = True)
    parser.add_argument('--gpu', help = "Set to run on GPU", action = 'store_true', default = False)
    parser.add_argument('--seed', help = "Set the random seed.", type = int, default = 32)
    parser.add_argument("--experiment", help = "Set experiment to run.", default = "word_token", type = str.lower)
    parser.add_argument("--slur_window", help = "Set window size for slur replacement.")

    args = parser.parse_args()

    if args.encoding == 'embedding':
        mod_lib = emb
        onehot = False
    elif args.encoding == 'onehot':
        mod_lib = oh
        onehot = True
    elif args.encoding in ['tfidf', 'count']:
        mod_lib = lin

    # Set seeds
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    c = Cleaner(args.cleaners)
    p = Preprocessors(args.datadir)

    # Word token experiment
    if args.experiment == 'word':
        # Set training dataset
        experiment = p.word_token

    elif args.experiment == 'liwc':
        experiment = p.compute_unigram_liwc

    elif args.experiment in ['ptb', 'pos']:
        experiment = p.ptb_tokenize

    elif args.experiment == 'length':
        experiment = p.word_length

    elif args.experiment == 'syllable':
        experiment = p.syllable_count

    elif args.experiment == 'slur':
        p.slur_window = args.slur_window
        experiment = p.slur_replacement

    if args.train == 'davidson':
        main = loaders.davidson(c, args.datadir, preprocessor = experiment,
                                label_processor = loaders.davidson_to_binary, stratify = 'label', skip_header = True)
        test_sets = [main,
                     loaders.davidson(c, args.datadir, preprocessor = experiment,
                                      label_processor = loaders.davidson_to_binary, stratify = 'label',
                                      skip_header = True),
                     loaders.wulczyn(c, args.datadir, preprocessor = experiment, stratify = 'label',
                                     skip_header = True),
                     loaders.garcia(c, args.datadir, preprocessor = experiment,
                                    label_processor = loaders.binarize_garcia, stratify = 'label', skip_header = True),
                     loaders.waseem(c, args.datadir, preprocessor = experiment,
                                    label_processor = loaders.waseem_to_binary, stratify = 'label'),
                     loaders.waseem_hovy(c, args.datadir, preprocessor = experiment,
                                         label_processor = loaders.waseem_to_binary,
                                         stratify = 'label')
                     ]

    elif args.train == 'waseem':
        main = loaders.waseem(c, args.datadir, preprocessor = experiment, label_processor = loaders.waseem_to_binary,
                              stratify = 'label'),
        test_sets = [main,
                     loaders.wulczyn(c, args.datadir, preprocessor = experiment, stratify = 'label',
                                     skip_header = True),
                     loaders.garcia(c, args.datadir, preprocessor = experiment,
                                    label_processor = loaders.binarize_garcia, stratify = 'label', skip_header = True),
                     loaders.davidson(c, args.datadir, preprocessor = experiment,
                                      label_processor = loaders.davidson_to_binary, stratify = 'label',
                                      skip_header = True),
                     loaders.waseem_hovy(c, args.datadir, preprocessor = experiment,
                                         label_processor = loaders.waseem_to_binary,
                                         stratify = 'label')
                     ]

    elif args.train == 'waseem_hovy':
        main = loaders.waseem_hovy(c, args.datadir, preprocessor = experiment,
                                   label_processor = loaders.waseem_to_binary,
                                   stratify = 'label')
        test_sets = [main,
                     loaders.davidson(c, args.datadir, preprocessor = experiment,
                                      label_processor = loaders.davidson_to_binary, stratify = 'label',
                                      skip_header = True),
                     loaders.wulczyn(c, args.datadir, preprocessor = experiment, stratify = 'label',
                                     skip_header = True),
                     loaders.garcia(c, args.datadir, preprocessor = experiment,
                                    label_processor = loaders.binarize_garcia,
                                    stratify = 'label', skip_header = True),
                     loaders.waseem(c, args.datadir, preprocessor = experiment,
                                    label_processor = loaders.waseem_to_binary,
                                    stratify = 'label')
                     ]

    elif args.train == 'garcia':
        main = loaders.garcia(c, args.datadir, preprocessor = experiment, label_processor = loaders.binarize_garcia,
                              stratify = 'label', skip_header = True),
        test_sets = [main,
                     loaders.davidson(c, args.datadir, preprocessor = experiment,
                                      label_processor = loaders.davidson_to_binary, stratify = 'label',
                                      skip_header = True),
                     loaders.wulczyn(c, args.datadir, preprocessor = experiment, stratify = 'label',
                                     skip_header = True),
                     loaders.waseem(c, args.datadir, preprocessor = experiment,
                                    label_processor = loaders.waseem_to_binary,
                                    stratify = 'label'),
                     loaders.waseem_hovy(c, args.datadir, preprocessor = experiment,
                                         label_processor = loaders.waseem_to_binary,
                                         stratify = 'label')
                     ]

    elif args.train == 'wulczyn':
        main = loaders.wulczyn(c, args.datadir, preprocessor = experiment, stratify = 'label', skip_header = True)
        test_sets = [main,
                     loaders.davidson(c, args.datadir, preprocessor = experiment,
                                      label_processor = loaders.davidson_to_binary, stratify = 'label',
                                      skip_header = True),
                     loaders.garcia(c, args.datadir, preprocessor = experiment,
                                    label_processor = loaders.binarize_garcia,
                                    stratify = 'label', skip_header = True),
                     loaders.waseem(c, args.datadir, preprocessor = experiment,
                                    label_processor = loaders.waseem_to_binary,
                                    stratify = 'label'),
                     loaders.waseem_hovy(c, args.datadir, preprocessor = experiment,
                                         label_processor = loaders.waseem_to_binary,
                                         stratify = 'label')
                     ]

    # Define arugmen dictionaries
    train_args = {}
    model_args = {}

    # Set models to iterate over
    models = []
    for m in args.model:
        if m == 'mlp':
            models.append(mod_lib.MLPClassifier)
        if m == 'lstm':
            models.append(mod_lib.LSTMClassifier)
        if m == 'cnn':
            models.append(mod_lib.CNNClassifier)
            train_args['window_sizes'] = args.window_sizes
            train_args['num_filters'] = args.filters
            train_args['max_feats'] = args.max_feats
        if m == 'rnn':
            models.append(mod_lib.RNNClassifier)
        if m == 'all':
            models = [mod_lib.MLPClassifier,
                      mod_lib.CNNClassifier,
                      mod_lib.LSTMClassifier,
                      mod_lib.RNNClassifier]

    main.build_token_vocab(main.data)
    main.build_label_vocab(main.data)

    # Open output files
    enc = 'a' if os.path.isfile(f'{args.results}/{args.encoding}_{args.experiments}_train.tsv') else 'w'
    pred_enc = 'a' if os.path.isfile(f'{args.results}/{args.encoding}_{args.experiments}_preds.tsv') else 'w'

    train_writer = csv.writer(open(f"{args.results}/{args.encoding}_{args.experiment}_train.tsv", enc,
                                   encoding = 'utf-8'), delimiter = '\t')
    test_writer = csv.writer(open(f"{args.results}/{args.encoding}_{args.experiment}_test.tsv", enc,
                                   encoding = 'utf-8'), delimiter = '\t')
    pred_writer = csv.writer(open(f"{args.results}/{args.encoding}_{args.experiment}_preds.tsv", pred_enc,
                                  encoding = 'utf-8'), delimiter = '\t')

    model_hdr = ['Model', 'Input dim', 'Embedding dim', 'Hidden dim', 'Output dim', 'Window Sizes', '# Filters',
                 '# Layers', 'Dropout', 'Activation']
    if enc == 'w':
        metric_hdr = args.metrics + ['loss'] + [f"dev {m}" for m in args.metrics] + ['dev loss']
        hdr = ['Timestamp', 'Trained on', 'Evaluated on', 'Batch size', '# Epochs', 'Learning Rate'] + model_hdr
        hdr += metric_hdr
        train_writer.writerow(hdr)
        test_writer.writerow(hdr)

    if pred_enc == 'w':
        metric_hdr = args.metrics + ['loss']
        hdr = ['Timestamp', 'Trained on', 'Evaluated on', 'Batch size', '# Epochs', 'Learning Rate'] + model_hdr
        hdr += metric_hdr
        pred_writer.writerow(hdr)

    train_args['model_hdr'] = model_hdr
    train_args['metric_hdr'] = args.metrics + ['loss']

    with tqdm(args.batch_size, desc = "Batch Size Iterator") as b_loop,\
         tqdm(args.dropout, desc = "Dropout Iterator") as d_loop,\
         tqdm(args.learning_rate, desc = "Learning Rate Iterator") as lr_loop,\
         tqdm(args.embedding, desc = "Embedding Size Iterator") as e_loop,\
         tqdm(args.hidden, desc = "Hidden Dim Iterator") as h_loop,\
         tqdm(args.epochs, desc = "Epoch Count Iterator") as ep_loop,\
         tqdm(models, desc = "Iterating Models") as m_loop:

        train_args = {'num_layers': 1,
                      'shuffle': args.shuffle,
                      'batch_first': True,
                      'gpu': args.gpu,
                      'save_path': f"{args.save_model}{args.experiment}_best",
                      'early_stopping': args.patience,
                      'low': True if args.stop_metric == 'loss' else False
                      }

        if args.optimizer == 'adam':
            model_args['optimizer'] = torch.optim.Adam
        elif args.optimizer == 'sgd':
            model_args['optimizer'] = torch.optim.SGD
        elif args.optimizer == 'asgd':
            model_args['optimizer'] = torch.optim.ASGD
        elif args.optimizer == 'adamw':
            model_args['optimizer'] = torch.optim.AdamW

        # Explains losses:
        # https://medium.com/udacity-pytorch-challengers/a-brief-overview-of-loss-functions-in-pytorch-c0ddb78068f7
        if args.loss == 'nlll':
            model_args['loss_func'] = torch.nn.NLLLoss
        elif args.loss == 'crossentropy':
            model_args['loss_func'] = torch.nn.CrossEntropyLoss

        # Set input and ouput dims
        train_args['input_dim'] = main.vocab_size()
        train_args['output_dim'] = main.label_count()
        train_args['main_name'] = main.name

        # Batch all evaluation datasets
        test_batches = [process_and_batch(main, data.test, args.batch_size, onehot) for data in test_sets]

        for epoch in ep_loop:
            train_args['epochs'] = epoch

            for e in e_loop:
                e_loop.set_postfix(emb_dim = e)
                train_args['embedding_dim'] = e

                for hidden in h_loop:
                    h_loop.set_postfix(hid_dim = hidden)
                    train_args['hidden_dim'] = hidden

                    for batch_size in b_loop:
                        b_loop.set_postfix(batch_size = batch_size)
                        train_args['batchers'] = process_and_batch(main, main.data, batch_size, onehot)
                        train_args['dev'] = process_and_batch(main, main.dev, batch_size, onehot)

                        for dropout in d_loop:
                            d_loop.set_postfix(dropout = dropout)
                            train_args['dropout'] = dropout

                            for lr in lr_loop:
                                lr_loop.set_postfix(learning_rate = lr)

                                # hyper_info = ['Batch size', '# Epochs', 'Learning Rate']
                                train_args['hyper_info'] = [batch_size, epoch, lr]

                                for model in m_loop:
                                    # Intialize model, loss, optimizer, and metrics
                                    train_args['model'] = model(**train_args) if not args.gpu else model(**train_args).cuda()
                                    train_args['loss'] = model_args['loss']()
                                    train_args['optimizer'] = model_args['optimizer'](model.parameters(), lr)
                                    train_args['metrics'] = Metrics(args.metrics, args.display, args.stop_metric)
                                    train_args['dev_metrics'] = Metrics(args.metrics, args.display, args.stop_metric)
                                    train_args['data_name'] = main.name
                                    m_loop.set_postfix(model = model.name)  # Update loop to contain name of model.

                                    run_model(train = True, writer = train_writer, **train_args)

                                    for batcher, test in zip(test_batches, test_sets):
                                        eval_args = {'model': train_args['model'],
                                                     'batchers': batcher,
                                                     'loss': train_args['loss'],
                                                     'metrics': Metrics(args.metrics, args.display, args.stop_metric),
                                                     'gpu': args.gpu,
                                                     'data': test,
                                                     'dataset': main,
                                                     'hyper_info': train_args['hyper_info'],
                                                     'model_hdr': train_args['model_hdr'],
                                                     'metric_hdr': train_args['metric_hdr'],
                                                     'main_name': train_args['main_name'],
                                                     'data_name': test.name,
                                                     'train_field': 'text',
                                                     'label_field': 'label'
                                                     }

                                        run_model(train = False, writer = test_writer, pred_writer = pred_writer,
                                                  **eval_args)
