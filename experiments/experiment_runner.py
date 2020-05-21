import os
import csv
import torch
import argparse
import numpy as np
from tqdm import tqdm
import mlearn.modeling.onehot as oh
import mlearn.data.loaders as loaders
import mlearn.modeling.embedding as emb
from mlearn.utils.metrics import Metrics
from mlearn.utils.pipeline import process_and_batch
from mlearn.data.clean import Cleaner, Preprocessors
from mlearn.utils.train import run_singletask_model as run_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run Experiments to generalise models.")

    # For all models
    parser.add_argument("--train", help = "Choose train data: davidson, Waseem, Waseem and Hovy, wulczyn, and garcia.")
    parser.add_argument("--model", help = "Choose the model to be run: CNN, RNN, LSTM, MLP, LR.", default = "mlp")
    parser.add_argument("--epochs", help = "Set the number of epochs.", default = 200, type = int)
    parser.add_argument("--batch_size", help = "Set the batch size.", default = 64, type = int)
    parser.add_argument("--save_model", help = "Directory to store models in.")
    parser.add_argument("--results", help = "Set file to output results to.")
    parser.add_argument("--cleaners", help = "Set the cleaning routines to be used.", nargs = '+', default = None)
    parser.add_argument("--metrics", help = "Set the metrics to be used.", nargs = '+', default = ["f1"], type = str)
    parser.add_argument("--display", help = "Metric to display in TQDM loops.", default = 'accuracy')
    parser.add_argument("--datadir", help = "Path to the datasets.", default = 'data/')

    # Model (hyper) parameters
    parser.add_argument("--embedding", help = "Set the embedding dimension.", default = 300, type = int)
    parser.add_argument("--hidden", help = "Set the hidden dimension.", default = 128, type = int)
    parser.add_argument("--layers", help = "Set the number of layers.", default = 1, type = int)
    parser.add_argument("--window_sizes", help = "Set the window sizes.", nargs = '+', default = [2, 3, 4], type = int)
    parser.add_argument("--filters", help = "Set the number of filters for CNN.", default = 128, type = int)
    parser.add_argument("--max_feats", help = "Set the number of features for CNN.", default = 100, type = int)
    parser.add_argument("--dropout", help = "Set value for dropout.", default = 0.0, type = float)
    parser.add_argument("--optimizer", help = "Optimizer to use.", default = 'adam')
    parser.add_argument("--loss", help = "Loss to use.", default = 'NLLL')
    parser.add_argument('--learning_rate', help = "Set the learning rate for the model.", default = 0.01, type = float)
    parser.add_argument('--gpu', help = "Set to run on GPU", action = 'store_true', default = False)
    parser.add_argument('--shuffle', help = "Shuffle dataset between epochs", action = 'store_true', default = True)
    parser.add_argument('--seed', help = "Set the random seed.", type = int, default = 32)
    parser.add_argument('--onehot', help = "Use one-hot tensors.", action = 'store_true', default = False)

    # Experiment parameters
    parser.add_argument("--experiment", help = "Set experiment to run.", default = "word_token")
    parser.add_argument("--slur_window", help = "Set window size for slur replacement.")

    args = parser.parse_args()

    models = oh if args.onehot else emb

    train_args = {'model': None,
                  'epochs': args.epochs,
                  'iterator': None,
                  'dev_iterator': None,
                  'loss_func': None,
                  'num_layers': 1,
                  'batch_first': True,
                  'metrics': Metrics(args.metrics, args.display),
                  'dev_metrics': Metrics(args.metrics, args.display),
                  'dropout': args.dropout,
                  'embedding_dim': args.embedding,
                  'hidden_dim': args.hidden,
                  'window_sizes': args.window_sizes,
                  'num_filters': args.filters,
                  'max_feats': args.max_feats,
                  'output_dim': None,
                  'input_dim': None,
                  'gpu': args.gpu,
                  'shuffle': args.shuffle,
                  'save_path': os.path.join(args.results, 'models/')
                  }

    eval_args = {'model': None,
                 'iterator': None,
                 'loss_func': None,
                 'metrics': train_args['metrics'],
                 'gpu': args.gpu
                 }

    # Set seeds
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    c = Cleaner(args.cleaners)
    p = Preprocessors(args.datadir)

    args.experiment = args.experiment.lower()
    args.train = args.train.lower()
    args.loss = args.loss.lower()
    args.optimizer = args.optimizer.lower()
    args.model = args.model.lower()

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
        experiement = p.slur_replacement

    if args.train == 'davidson':
        main = loaders.davidson(c, args.datadir, preprocessor = experiment,
                                label_processor = loaders.davidson_to_binary, stratify = 'label', skip_header = True)
        evals = [main,
                 loaders.davidson(c, args.datadir, preprocessor = experiment,
                                  label_processor = loaders.davidson_to_binary, stratify = 'label', skip_header = True),
                 loaders.wulczyn(c, args.datadir, preprocessor = experiment, stratify = 'label', skip_header = True),
                 loaders.garcia(c, args.datadir, preprocessor = experiment, label_processor = loaders.binarize_garcia,
                                stratify = 'label', skip_header = True),
                 loaders.waseem(c, args.datadir, preprocessor = experiment, label_processor = loaders.waseem_to_binary,
                                stratify = 'label'),
                 loaders.waseem_hovy(c, args.datadir, preprocessor = experiment,
                                     label_processor = loaders.waseem_to_binary,
                                     stratify = 'label')
                 ]

    elif args.train == 'waseem':
        main = loaders.waseem(c, args.datadir, preprocessor = experiment, label_processor = loaders.waseem_to_binary,
                              stratify = 'label'),
        evals = [main,
                 loaders.wulczyn(c, args.datadir, preprocessor = experiment, stratify = 'label', skip_header = True),
                 loaders.garcia(c, args.datadir, preprocessor = experiment, label_processor = loaders.binarize_garcia,
                                stratify = 'label', skip_header = True),
                 loaders.davidson(c, args.datadir, preprocessor = experiment,
                                  label_processor = loaders.davidson_to_binary, stratify = 'label', skip_header = True),
                 loaders.waseem_hovy(c, args.datadir, preprocessor = experiment,
                                     label_processor = loaders.waseem_to_binary,
                                     stratify = 'label')
                 ]

    elif args.train == 'waseem_hovy':
        main = loaders.waseem_hovy(c, args.datadir, preprocessor = experiment,
                                   label_processor = loaders.waseem_to_binary,
                                   stratify = 'label')
        evals = [main,
                 loaders.davidson(c, args.datadir, preprocessor = experiment,
                                  label_processor = loaders.davidson_to_binary, stratify = 'label', skip_header = True),
                 loaders.wulczyn(c, args.datadir, preprocessor = experiment, stratify = 'label', skip_header = True),
                 loaders.garcia(c, args.datadir, preprocessor = experiment, label_processor = loaders.binarize_garcia,
                                stratify = 'label', skip_header = True),
                 loaders.waseem(c, args.datadir, preprocessor = experiment, label_processor = loaders.waseem_to_binary,
                                stratify = 'label')
                 ]

    elif args.train == 'garcia':
        main = loaders.garcia(c, args.datadir, preprocessor = experiment, label_processor = loaders.binarize_garcia,
                              stratify = 'label', skip_header = True),
        evals = [main,
                 loaders.davidson(c, args.datadir, preprocessor = experiment,
                                  label_processor = loaders.davidson_to_binary, stratify = 'label', skip_header = True),
                 loaders.wulczyn(c, args.datadir, preprocessor = experiment, stratify = 'label', skip_header = True),
                 loaders.waseem(c, args.datadir, preprocessor = experiment, label_processor = loaders.waseem_to_binary,
                                stratify = 'label'),
                 loaders.waseem_hovy(c, args.datadir, preprocessor = experiment,
                                     label_processor = loaders.waseem_to_binary,
                                     stratify = 'label')
                 ]

    elif args.train == 'wulczyn':
        main = loaders.wulczyn(c, args.datadir, preprocessor = experiment, stratify = 'label', skip_header = True)
        evals = [main,
                 loaders.davidson(c, args.datadir, preprocessor = experiment,
                                  label_processor = loaders.davidson_to_binary, stratify = 'label', skip_header = True),
                 loaders.garcia(c, args.datadir, preprocessor = experiment, label_processor = loaders.binarize_garcia,
                                stratify = 'label', skip_header = True),
                 loaders.waseem(c, args.datadir, preprocessor = experiment, label_processor = loaders.waseem_to_binary,
                                stratify = 'label'),
                 loaders.waseem_hovy(c, args.datadir, preprocessor = experiment,
                                     label_processor = loaders.waseem_to_binary,
                                     stratify = 'label')
                 ]

    main.build_token_vocab(main.data)
    main.build_label_vocab(main.data)

    train_args['input_dim'] = main.vocab_size()
    train_args['output_dim'] = main.label_count()
    train_args['gpu'] = args.gpu

    model_args = {}

    if args.optimizer == 'adam':
        model_args['optimizer'] = torch.optim.Adam
    elif args.optimizer == 'sgd':
        model_args['optimizer'] = torch.optim.SGD
    elif args.optimizer == 'asgd':
        model_args['optimizer'] = torch.optim.ASGD
    elif args.optimizer == 'adamw':
        model_args['optimizer'] = torch.optim.AdamW

    if args.loss == 'nlll':
        model_args['loss_func'] = torch.nn.NLLLoss
    elif args.loss == 'crossentropy':
        model_args['loss_func'] = torch.nn.CrossEntropyLoss

    # Set models
    if args.model == 'mlp':
        models = [models.MLPClassifier(**train_args)]
        model_header = ['epoch', 'model', 'input dim', 'embedding dim', 'hidden dim', 'output dim', 'dropout',
                        'learning rate']
        model_info = {'mlp': ['mlp', train_args['input_dim'], train_args['embedding_dim'], train_args['hidden_dim'],
                              train_args['output_dim'], train_args['dropout'], args.learning_rate]}

    elif args.model == 'lstm':
        models = [models.LSTMClassifier(**train_args)]
        model_header = ['epoch', 'model', 'input dim', 'embedding dim', 'hidden dim', 'output dim', 'num layers',
                        'learning rate']
        model_info = {'lstm': ['lstm', train_args['input_dim'], train_args['embedding_dim'], train_args['hidden_dim'],
                               train_args['output_dim'], train_args['num_layers'], args.learning_rate]}

    elif args.model == 'rnn':
        models = [models.RNNClassifier(**train_args)]
        model_header = ['epoch', 'model', 'input dim', 'embedding dim', 'hidden dim', 'output dim', 'num layers',
                        'learning rate']
        model_info = {'rnn': ['rnn', train_args['input_dim'], train_args['embedding_dim'], train_args['hidden_dim'],
                              train_args['output_dim'], train_args['num_layers'], args.learning_rate]}

    elif args.model == 'cnn':
        # Make sure all documents adhere to the maximum number of features.
        args.onehot = False
        for d in evals:
            d.modify_length = args.max_feats

        models = [models.CNNClassifier(**train_args)]
        model_header = ['epoch', 'model', 'window sizes', 'num filters', 'max feats', 'hidden dim', 'output dim']
        model_info = {'cnn': ['cnn', train_args['window_sizes'], train_args['num_filters'], train_args['max_feats'],
                              train_args['hidden_dim'], train_args['output_dim']]}

    elif args.model == 'all':
        model_header = ['epoch', 'model', 'input dim', 'hidden dim', 'embedding dim', 'dropout', 'learning rate',
                        'window sizes', 'num filters', 'max feats', 'output dim']
        model_info = {'all': [train_args['input_dim'], train_args['hidden_dim'], train_args['embedding_dim'],
                              train_args['dropout'], args.learning_rate, train_args['window_sizes'],
                              train_args['num_filters'], train_args['max_feats'], train_args['output_dim']]}

        models = [models.MLPClassifier(**train_args),
                  models.CNNClassifier(**train_args),
                  models.LSTMClassifier(**train_args),
                  models.RNNClassifier(**train_args)]

    train_args['iterator'] = process_and_batch(main, main.data, args.batch_size, args.onehot)

    if main.dev is not None:  # As the dataloaders always create a dev set, this condition will always be True
        train_args['dev_iterator'] = process_and_batch(main, main.dev, args.batch_size, args.onehot)

    test_sets = [process_and_batch(main, data.test, args.batch_size, args.onehot) for data in evals]

    # Initialize writers
    # TODO Add experiment name to each output file.
    enc = 'a' if os.path.isfile(args.results + '_train') else 'w'
    with open(args.results + '_train', enc, encoding = 'utf-8') as train_res,\
            open(args.results + '_test', enc, encoding = 'utf-8') as test_res:

        train_writer = csv.writer(train_res, delimiter = '\t')
        test_writer = csv.writer(test_res, delimiter = '\t')

        # Create header
        metrics = list(train_args['metrics'].list())
        train_header = ['dataset', 'trained on'] + model_header + metrics + ['train loss']
        train_header += ['dev ' + m for m in metrics] + ['dev loss']
        test_header = ['dataset', 'trained on'] + model_header + metrics + ['loss']

        if enc == 'w':  # Only write headers if the file doesn't already exist.
            train_writer.writerow(train_header)
            test_writer.writerow(test_header)

        for model in tqdm(models, desc = "Iterating over models"):
            train_args['model'] = model if not args.gpu else model.cuda()

            if args.model == 'all':
                info = [model.name] + model_info['all']
            else:
                info = model_info[model.name]

            # Explains losses:
            # https://medium.com/udacity-pytorch-challengers/a-brief-overview-of-loss-functions-in-pytorch-c0ddb78068f7
            train_args['loss_func'] = model_args['loss_func']()
            train_args['optimizer'] = model_args['optimizer'](model.parameters(), args.learning_rate)
            train_args['data_name'] = main.name
            train_args['main_name'] = main.name

            run_model('pytorch', train = True, writer = train_writer, model_info = info, head_len = len(train_header),
                      **train_args)

            for data, iterator in tqdm(zip(evals, test_sets), desc = 'Evaluate', leave = False, total = len(evals)):
                # Test on other datasets.
                # Process and batch the data
                eval_args['iterator'] = iterator
                eval_args['data_name'] = data.name
                eval_args['main_name'] = main.name
                eval_args['epochs'] = 1

                # Set up the model arguments
                eval_args['model'] = model
                eval_args['loss_func'] = train_args['loss_func']

                # Run the model
                run_model('pytorch', train = False, writer = test_writer, model_info = info,
                          head_len = len(test_header), **eval_args)
