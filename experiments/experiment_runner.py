import sys
import torch
import argparse
sys.path.extend(['/Users/zeerakw/PhD/projects/active/Generalisable_abuse/'])
from gen.shared import base
from gen.shared.train import run_model
import gen.shared.dataloaders as loaders
from gen.shared.metrics import select_metrics
from gen.shared.clean import Cleaner, Preprocessors
from gen.shared.batching import Batch, BatchExtractor
from gen.neural import CNNClassifier, MLPClassifier, RNNClassifier, LSTMClassifier


def process_and_batch(dataset, data, batch_size):
    """Process a dataset and data.
    :dataset: A dataset object.
    :data: Data to be processed.
    :returns: Processed data.
    """
    # Process labels and encode data.
    dataset.process_labels(data)
    encoded = dataset.encode(data, onehot = True)

    # Batch data
    batch = Batch(batch_size, encoded)
    batch.create_batches()
    batches = BatchExtractor('encoded', 'label', batch)

    return batches


def split_data(dataset, split_ratio: base.Union[base.List[float], float]):
    """Load dataaset and split it.
    :dataset: Dataset object to be loaded.
    :split_ratio (base.Union[base.List[float], float]): Float or list of floats.
    :returns: TODO
    """
    if dataset.test is None:
        dataset.split(dataset.data, split_ratio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run Experiments to generalise models.")

    # For all models
    parser.add_argument("--train", help = "Choose train data: davidson, Waseem, Waseem and Hovy, wulczyn, and garcia.")
    parser.add_argument("--binary", help = "Reduce labels to binary classification", action = "store_true")
    parser.add_argument("--model", help = "Choose the model to be run: CNN, RNN, LSTM, MLP, LR.", default = "mlp")
    parser.add_argument("--epochs", help = "Set the number of epochs.", default = 200)
    parser.add_argument("--batch_size", help = "Set the batch size.", default = 200)
    parser.add_argument("--save_model", help = "Directory to store models in.")
    parser.add_argument("--results", help = "Set file to output results to.")
    parser.add_argument("--cleaners", help = "Set the cleaning routines to be used.", nargs = '+', default = None)
    parser.add_argument("--metrics", help = "Set the metrics to be used.", default = "f1")

    # Model (hyper) parameters
    parser.add_argument("--embedding", help = "Set the embedding dimension.", default = 300)
    parser.add_argument("--hidden", help = "Set the hidden dimension.", default = 128)
    parser.add_argument("--window_sizes", help = "Set the window sizes.", nargs = '+', default = [2, 3, 4])
    parser.add_argument("--filters", help = "Set the number of filters for CNN.", default = 128)
    parser.add_argument("--max_feats", help = "Set the number of features for CNN.", default = 100)
    parser.add_argument("--dropout", help = "Set value for dropout.", default = 0.2)
    parser.add_argument("--optimizer", help = "Optimizer to use.", default = 'adam')
    parser.add_argument("--loss", help = "Loss to use.", default = 'NLLL')
    parser.add_argument('--learning_rate', help = "Set the learning rate for the model.", default = 0.01)

    # Experiment parameters
    parser.add_argument("--experiment", help = "Set experiment to run.", default = "word_token")
    parser.add_argument("--task", help = "Set experiment to run.", default = "classification")

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

    __import__('pdb').set_trace()
    main.build_token_vocab(main.data)
    main.build_label_vocab(main.data)

    train_args['input_dim'] = main.vocab_size()
    train_args['output_dim'] = main.label_count()
    train_args['batches'] = process_and_batch(main, main.data)

    if main.test is not None:
        train_args['dev_batches'] = process_and_batch(main, main.test, args.batch_size)
    else:
        train_args['dev_batches'] = process_and_batch(main, main.dev, args.batch_size)

    # Set models
    if args.model == 'mlp':
        models = [MLPClassifier(**train_args)]

    elif args.model == 'cnn':
        models = [CNNClassifier(**train_args)]

    elif args.model == 'lstm':
        models = [LSTMClassifier(**train_args)]

    elif args.model == 'rnn':
        models = [RNNClassifier(**train_args)]

    elif args.model == 'all':
        models = [MLPClassifier(**train_args),
                  CNNClassifier(**train_args),
                  LSTMClassifier(**train_args),
                  RNNClassifier(**train_args)]

    for model in models:
        # Explains losses:
        # https://medium.com/udacity-pytorch-challengers/a-brief-overview-of-loss-functions-in-pytorch-c0ddb78068f7
        train_args['loss_func'] = torch.nn.NLLLoss() if args.loss.lower() == 'nlll' else torch.nn.CrossEntropyLoss()
        train_args['optimizer'] = torch.optim.adam(model.parameters(), args.learning_rate)

        run_model('pytorch', train = True, **train_args)

        for data in others:  # Test on other datasets.
            # Split the data
            split_data(data)

            # Process and batch the data
            batched = process_and_batch(main, data.test, args.batch_size)

            # Set up the model arguments
            eval_args['model'] = model
            eval_args['loss_func'] = train_args['loss_func']

            # Run the model
            run_model('pytorch', train = False, **eval_args)
