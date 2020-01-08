import sys
import csv
import torch
import argparse
sys.path.extend(['/Users/zeerakw/PhD/projects/active/Generalisable_abuse/'])
from gen.shared.train import run_model, process_and_batch
import gen.shared.dataloaders as loaders
from gen.shared.metrics import select_metrics
from gen.shared.clean import Cleaner, Preprocessors
from gen.neural import CNNClassifier, MLPClassifier, RNNClassifier, LSTMClassifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run Experiments to generalise models.")

    # For all models
    parser.add_argument("--train", help = "Choose train data: davidson, Waseem, Waseem and Hovy, wulczyn, and garcia.")
    parser.add_argument("--model", help = "Choose the model to be run: CNN, RNN, LSTM, MLP, LR.", default = "mlp")
    parser.add_argument("--epochs", help = "Set the number of epochs.", default = 200, type = int)
    parser.add_argument("--batch_size", help = "Set the batch size.", default = 200, type = int)
    parser.add_argument("--save_model", help = "Directory to store models in.")
    parser.add_argument("--results", help = "Set file to output results to.")
    parser.add_argument("--cleaners", help = "Set the cleaning routines to be used.", nargs = '+', default = None)
    parser.add_argument("--metrics", help = "Set the metrics to be used.", default = ["f1"], type = str)
    parser.add_argument("--display", help = "Metric to display in TQDM loops.")

    # Model (hyper) parameters
    parser.add_argument("--embedding", help = "Set the embedding dimension.", default = 300, type = int)
    parser.add_argument("--hidden", help = "Set the hidden dimension.", default = 128, type = int)
    parser.add_argument("--layers", help = "Set the number of layers.", default = 1, type = int)
    parser.add_argument("--window_sizes", help = "Set the window sizes.", nargs = '+', default = [2, 3, 4], type = int)
    parser.add_argument("--filters", help = "Set the number of filters for CNN.", default = 128, type = int)
    parser.add_argument("--max_feats", help = "Set the number of features for CNN.", default = 100, type = int)
    parser.add_argument("--dropout", help = "Set value for dropout.", default = 0.2, type = float)
    parser.add_argument("--optimizer", help = "Optimizer to use.", default = 'adam')
    parser.add_argument("--loss", help = "Loss to use.", default = 'NLLL')
    parser.add_argument('--learning_rate', help = "Set the learning rate for the model.", default = 0.01, type = float)

    # Experiment parameters
    parser.add_argument("--experiment", help = "Set experiment to run.", default = "word_token")
    parser.add_argument("--slur_window", help = "Set window size for slur replacement.")

    args = parser.parse_args()

    train_args = {'model': None,
                  'epochs': args.epochs,
                  'batches': None,
                  'dev_batches': None,
                  'loss_func': None,
                  'num_layers': 1,
                  'batch_first': True,
                  'metrics': select_metrics(args.metrics),
                  'dropout': args.dropout,
                  'embedding_dim': args.embedding,
                  'hidden_dim': args.hidden,
                  'window_sizes': args.window_sizes,
                  'num_filters': args.filters,
                  'max_feats': args.max_feats,
                  'output_dim': None,
                  'input_dim': None,
                  }

    eval_args = {'model': None,
                 'iterator': None,
                 'loss_func': None,
                 'metrics': None
                 }
    c = Cleaner(args.cleaners)
    p = Preprocessors()

    # Word token experiment
    args.experiment = args.experiment.lower()
    args.train = args.train.lower()
    args.loss = args.loss.lower()
    args.optimizer = args.optimizer.lower()
    args.model = args.model.lower()

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
        main = loaders.davidson(c, experiment)
        others = [loaders.wulczyn(c, experiment), loaders.garcia(c, experiment), loaders.waseem(c, experiment),
                  loaders.waseem_hovy(c, experiment)]

    elif args.train == 'waseem':
        main = loaders.waseem(c, experiment)
        others = [loaders.wulczyn(c, experiment), loaders.garcia(c, experiment), loaders.davidson(c, experiment),
                  loaders.waseem_hovy(c, experiment)]

    elif args.train == 'waseem_hovy':
        main = loaders.waseem_hovy(c, experiment)
        others = [loaders.wulczyn(c, experiment), loaders.garcia(c, experiment), loaders.davidson(c, experiment),
                  loaders.waseem(c, experiment)]

    elif args.train == 'garcia':
        main = loaders.garcia(c, experiment)
        others = [loaders.wulczyn(c, experiment), loaders.davidson(c, experiment), loaders.waseem_hovy(c, experiment),
                  loaders.waseem(c, experiment)]

    elif args.train == 'wulczyn':
        main = loaders.wulczyn(c, experiment)
        others = [loaders.garcia(c, experiment), loaders.davidson(c, experiment), loaders.waseem_hovy(c, experiment),
                  loaders.waseem(c, experiment)]

    main.build_token_vocab(main.data)
    main.build_label_vocab(main.data)

    train_args['input_dim'] = main.vocab_size()
    train_args['output_dim'] = main.label_count()
    train_args['batches'] = process_and_batch(main, main.data, args.batch_size)

    if args.optimizer == 'adam':
        train_args['optimizer'] = torch.optim.Adam
    elif args.optimizer == 'sgd':
        train_args['optimizer'] = torch.optim.SGD
    elif args.optimizer == 'asgd':
        train_args['optimizer'] = torch.optim.ASGD
    elif args.optimizer == 'adamw':
        train_args['optimizer'] = torch.optim.AdamW

    if args.loss == 'nlll':
        train_args['loss_func'] = torch.nn.NLLLoss
    elif args.loss == 'crossentropy':
        train_args['loss_func'] = torch.nn.CrossEntropyLoss

    if main.test is not None:
        train_args['dev_batches'] = process_and_batch(main, main.test, args.batch_size)
    else:
        train_args['dev_batches'] = process_and_batch(main, main.dev, args.batch_size)

    # Set models
    if args.model == 'mlp':
        models = [MLPClassifier(**train_args)]
        model_header = ['epoch', 'model', 'input dim', 'embedding dim', 'hidden dim', 'output', 'dropout']
        model_info = {'mlp': ['mlp', train_args['input_dim'], train_args['embedding_dim'], train_args['hidden_dim'],
                              train_args['output_dim'], train_args['dropout']]}

    elif args.model == 'lstm':
        models = [LSTMClassifier(**train_args)]
        model_header = ['epoch', 'model', 'input dim', 'embedding dim', 'hidden dim', 'output', 'num layers']
        model_info = {'lstm': ['lstm', train_args['input_dim'], train_args['embedding_dim'], train_args['hidden_dim'],
                               train_args['output_dim'], train_args['num_layers']]}

    elif args.model == 'rnn':
        models = [RNNClassifier(**train_args)]
        model_header = ['epoch', 'model', 'input dim', 'embedding dim', 'hidden dim', 'output']
        model_info = {'rnn': ['rnn', train_args['input_dim'], train_args['embedding_dim'], train_args['hidden_dim'],
                              train_args['output_dim'], train_args['num_layers']]}

    # elif args.model == 'cnn':
    #     models = [CNNClassifier(**train_args)]
    #     model_header = ['epoch', 'model', 'window sizes', 'num filters', 'max feats', 'hidden dim', 'output dim']
    #     model_info = {'cnn': ['cnn', train_args['window_sizes'], train_args['num_filters'], train_args['max_feats'],
    #                           train_args['hidden_dim'], train_args['output_dim']]}

    elif args.model == 'all':
        model_header = ['epoch', 'model', 'input dim', 'hidden dim', 'embedding dim', 'dropout', 'window sizes',
                        'num filters', 'max feats', 'output dim']
        model_info = {'all': [train_args['input_dim'], train_args['hidden_dim'], train_args['embedding_dim'],
                              train_args['dropout'], train_args['window_sizes'], train_args['num_filters'],
                              train_args['max_feats'], train_args['output_dim']]}

        models = [MLPClassifier(**train_args),
                  # CNNClassifier(**train_args),
                  LSTMClassifier(**train_args),
                  RNNClassifier(**train_args)]

    # Initialize writers
    train_writer = csv.writer(open(args.results + '_train', 'a', encoding = 'utf-8'), delimiter = '\t')
    test_writer = csv.writer(open(args.results + '_test', 'a', encoding = 'utf-8'), delimiter = '\t')

    # Create header
    metrics = list(train_args['metrics'].keys())
    header = ['dataset'] + model_header + metrics + ['train loss'] + ['dev ' + m for m in metrics] + ['dev loss']
    train_writer.writerow(header)
    test_writer.writerow(header)

    for model in models:
        train_args['model'] = model

        if args.model == 'all':
            info = model_info['all']
        else:
            info = model_info[model.name]

        # Explains losses:
        # https://medium.com/udacity-pytorch-challengers/a-brief-overview-of-loss-functions-in-pytorch-c0ddb78068f7
        train_args['loss_func'] = train_args['loss_func']()
        train_args['optimizer'] = train_args['optimizer'](model.parameters(), args.learning_rate)

        run_model('pytorch', train = True, writer = train_writer, model_info = info, head_len = len(header),
                  **train_args)

        for data in others:  # Test on other datasets.
            # Process and batch the data
            batched = process_and_batch(main, data.test, args.batch_size)

            # Set up the model arguments
            eval_args['model'] = model
            eval_args['loss_func'] = train_args['loss_func']

            # Run the model
            run_model('pytorch', train = False, **eval_args)
