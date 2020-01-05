import sys
import torch.nn as nn
import torch.optim as optim
from ..gen.shared.base import Field
from ..gen.shared.clean import Cleaner
from ..gen.shared.data import GeneralDataset
from ..gen.shared.train import run_model
from ..gen.shared.batching import Batch, BatchExtractor


def experiment(args, train: base.DataType, others: base.List[base.DataType]):
    """Set up word token experiment.
    :args: Parsed commandline arguments.
    :train (base.DataType): The loaded dataset.
    :others (base.List[base.DataType]): List of
    :returns: TODO
    """
    # TODO Build dataset loaders first otherwise this won't make sense.
    # TODO The dataloaders must allow for different types of experiments
    # TODO That is they must take preprocessors in.

    # Build vocab for training task
    dataset.build_token_vocab(train.data)
    dataset.build_label_vocab(train.data)

    # Build


# FOR ALL DATASETS
clean = Cleaner(['lower', 'url', 'hashtag', 'username'])
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
OUTPUT_DIM = NO_CLASSES = 2
WINDOW_SIZES = [2, 3, 4]
NUM_FILTERS = 100
MAX_FEATS = 31
METRICS = ['precision', 'recall', 'f1']
training_args = {'model': None,
                 'epochs': 300,
                 'batches': None,
                 'dev_batches': None,
                 'loss_func': None,
                 'optimizer': None,
                 'metrics': METRICS
                 }
ignore = Field('ignore', train = False, label = False, ignore = True)


# DAVIDSON
def davidson_processor(label: str) -> int:
    """Reduce labels from from offensive & hate speech to offensive + hate speech.
    :label (str): label as a string.
    :returns: index.
    """
    if label == '0':
        return 0
    elif label == '1':
        return 0
    elif label == '2':
        return 1


# Fields
d_text = Field('text', train = True, label = False, ignore = False, ix = 6, cname = 'text')
d_label = Field('label', train = False, label = True, cname = 'label', ignore = False, ix = 5)
davidson_fields = [ignore, ignore, ignore, ignore, ignore, d_label, d_text]

# Get the data
davidson = GeneralDataset(data_dir = '~/PhD/projects/active/Generalisable_abuse/data/',
                          ftype = 'csv', fields = davidson_fields, train = 'davidson_train.csv', dev = None,
                          test = None, train_labels = None, tokenizer = clean.tokenize,
                          preprocessor = None, transformations = None,
                          label_processor = davidson_processor, sep = ',')
davidson.load('train')
davidson_train, davidson_dev = davidson.split(davidson.data, 0.8)  # Split into test and train

# Build vocabs and process labels
davidson.build_token_vocab(davidson_train)
davidson.build_label_vocab(davidson_train)
davidson.process_labels(davidson_train)
davidson.process_labels(davidson_dev)

# Encode dataset
davidson_train = davidson.encode(davidson_train, onehot = True)
davidson_dev = davidson.encode(davidson_dev, onehot = True)

batched = Batch(64, davidson_train)
batched.create_batches()
batched_train = BatchExtractor('encoded', 'label', batched)

batched = Batch(64, davidson_dev)
batched.create_batches()
batched_dev = BatchExtractor('encoded', 'label', batched)

D_VOCAB_SIZE = len(davidson.stoi)
lstm = LSTMClassifier(D_VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NO_CLASSES, no_layers = 1, batch_first = True)
rnn = RNNClassifier(D_VOCAB_SIZE, HIDDEN_DIM, OUTPUT_DIM, batch_first = True)
mlp = MLPClassifier(D_VOCAB_SIZE, HIDDEN_DIM, OUTPUT_DIM, dropout = 0.2, batch_first = True)
cnn = CNNClassifier(WINDOW_SIZES, NUM_FILTERS, MAX_FEATS, HIDDEN_DIM, NO_CLASSES, batch_first = True)
models = [rnn, lstm, mlp, cnn]

for model in models:
    print(f"Running {model.name}")
    training_args['model'] = model
    training_args['batches'] = batched_train
    training_args['dev_batches'] = batched_dev
    training_args['optimizer'] = optim.Adam(model.parameters(), lr = 0.01)
    training_args['loss_func'] = nn.NLLLoss()

    run_model('pytorch', train = True, **training_args)

# EVALUATE ON OTHER DATASETS


# WASEEM (Twitter)
def waseem_label_preprocess(label: str) -> int:
    """Preprocess label from raw waseem labels to binary.
    :returns (int): Processed label.
    """
    if label.lower() in ['racism', 'sexism', 'both']:
        return 0
    else:  # label == 'neither'
        return 1


w_text = Field('text', train = True, label = False, ignore = False, cname = 'text')
w_label = Field('label', train = False, label = True, ignore = False, cname = 'Annotation')
waseem_fields = [w_text, w_label]

waseem = GeneralDataset(data_dir = '~/PhD/projects/active/Generalisable_abuse/data/', ftype = 'json',
                        fields = waseem_fields, train = 'Wamateur_expert.json', dev = None, test = None,
                        train_labels = None, test_labels = None, dev_labels = None, tokenizer = clean.tokenize,
                        preprocessor = None, transformations = None, length = None,
                        label_processor = waseem_label_preprocess)
waseem.load('train')

# Build vocabs and process labels
waseem.build_token_vocab(waseem.data)
waseem.build_label_vocab(waseem.data)  # TODO This will be the old label mapping, not the modified mapping.
waseem.process_labels(waseem.data)

# WASEEM-HOVY (Twitter)

w_hovy = GeneralDataset(data_dir = '~/PhD/projects/active/Generalisable_abuse/data/', ftype = 'json',
                        fields = waseem_fields, train = 'Wamateur_expert.json', dev = None, test = None,
                        train_labels = None, test_labels = None, dev_labels = None, tokenizer = clean.tokenize,
                        preprocessor = None, transformations = None, length = None,
                        label_processor = waseem_label_preprocess)

# WULCZYN ET AL (Wikipedia)

# GARCIA ET AL (Stormfront)
