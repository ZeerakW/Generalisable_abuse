import os
import csv
import json
import torch
import numpy as np
from math import floor
import gen.shared.custom_types as t
from collections import Counter, defaultdict
from torch.utils.data import IterableDataset


class OnehotBatchGenerator:
    """A class to get the information from the batches."""

    def __init__(self, dataloader, datafield, labelfield, vocab_size):
        self.data, self.df, self.lf = dataloader, datafield, labelfield
        self.VOCAB_SIZE = vocab_size

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.data:
            X = torch.nn.functional.one_hot(getattr(batch, self.df), self.VOCAB_SIZE)
            y = getattr(batch, self.lf)
            yield (X, y)


class BatchGenerator:
    """A class to get the information from the batches."""

    def __init__(self, dataloader, datafield, labelfield):
        self.data, self.df, self.lf = dataloader, datafield, labelfield

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.data:
            X = getattr(batch, self.df)
            y = getattr(batch, self.lf)
            yield (X, y)


class Batch(object):
    """Create batches."""

    def __init__(self, data_attr, label_attr, batch_size, data):
        self.data_attr = data_attr
        self.label_attr = label_attr
        self.batch_size = batch_size
        self.data = data

    def create_batches(self):
        """Go over the data and create batches."""
        self.batches = []
        batch = []
        start_ix, end_ix = 0, self.batch_size
        for i in range(0, len(self.data), self.batch_size):
            batch = self.data[start_ix:end_ix]
            start_ix, end_ix = start_ix + self.batch_size, end_ix + self.batch_size
            self.batches.append(batch)

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        return self.batches[i]


class Field(object):
    """A class to set different properties of the individual fields."""
    def __init__(self, name: str, train: bool, label: bool, ignore: bool = True, ix: int = None,
                 cname: str = None):
        """Initialize the field object. Each individual field is to hold information about that field only.
        :param name (str): Name of the field.
        :param train (bool): Use for training.
        :param label (bool): Indicate if it is a label field.
        :param ignore (bool): Indicate whether to ignore the information in this field.
        :param ix (int, default = None): Index of the field in the splitted file. Only set this for [C|T]SV files.
        :param cname (str, default = None): Name of the column (/field) in the file. Only set this for JSON objects.
        Example Use:
            train_field = Field('text', train = True, label = False, ignore = False, ix = 0)
        """
        self.name = name
        self.train = train
        self.cname = cname
        self.label = label
        self.ignore = ignore
        self.index = ix


class Datapoint(object):
    """A class for each datapoint to instantiated as an object, which can allow for getting and setting attributes."""
    pass


class GeneralDataset(IterableDataset):
    """A general dataset class, which loads a dataset, creates a vocabulary, pads, tensorizes, etc."""
    def __init__(self, data_dir: str, ftype: str, sep: str, fields: t.List[Field],
                 train: str, dev: str = None, test: str = None, train_labels: str = None, dev_labels: str = None,
                 test_labels: str = None, tokenizer: t.Union[t.Callable, str] = 'spacy', lower: bool = True,
                 preprocessor: t.Callable = None, transformations: t.Callable = None,
                 label_processor: t.Callable = None):
        """Initialize the variables required for the dataset loading.
        :param data_dir (str): Path of the directory containing the files.
        :param ftype (str): ftype of the file ([C|T]SV and JSON accepted)
        :param sep (str): Separator token.
        :param fields (t.List[t.Tuple[str, ...]]): The names of the fields in the same order as they appear (in csv).
                    Example: ('data', None)
        :param train (str): Path to training file.
        :param dev (str, default None): Path to dev file, if dev file exists.
        :param test (str, default = None): Path to test file, if test file exists.
        :param train_labels (str, default = None): Path to file containing labels for training data.
        :param dev_labels (str, default = None): Path to file containing labels for dev data.
        :param test_labels (str, default = None): Path to file containing labels for test data.
        :param tokenizer (t.Callable or str, default = 'spacy'): Tokenizer to apply.
        :param lower (bool, default = True): Lowercase the document before tokenization.
        :param preprocessor (t.Callable, default = None): Preprocessing step to apply.
        :param transformations (t.Callable, default = None): Method changing from one representation to another.
        :param label_processor(t.Callable, default = None): Function to process labels with.
        """
        self.data_dir = os.path.abspath(data_dir) if '~' not in data_dir else os.path.expanduser(data_dir)

        try:
            ftype = ftype.upper()
            assert ftype in ['JSON', 'CSV', 'TSV']
            self.ftype = ftype
        except AssertionError as e:
            raise AssertionError("Input the correct file ftype: CSV/TSV or JSON")

        assert([getattr(f, 'label') is not None for f in fields])

        self.sep = sep
        self.fields = fields
        self.fields_dict = defaultdict(list)

        for field in self.fields:
            for key in field.__dict__:
                self.fields_dict[key].append(getattr(field, key))

        self.train_fields = [f for f in self.fields if f.train]
        self.label_fields = [f for f in self.fields if f.label]
        self.data_files = {key: os.path.join(self.data_dir, f) for f, key in zip([train, dev, test],
                                                                                 ['train', 'dev', 'test'])
                                                                                 if f is not None}
        self.label_files = {key: os.path.join(self.data_dir, f) for f, key in
                            zip([train_labels, dev_labels, test_labels], ['train', 'dev', 'test']) if f is not None}

        self.tokenizer = tokenizer
        self.lower = lower
        self.preprocessor = preprocessor
        self.data_dir = data_dir
        self.repr_transform = transformations
        self.label_processor = label_processor if label_processor else self.label_name_lookup
        self.length = 0

    def load(self, dataset: str = 'train', skip_header = True):
        """Load the dataset.
        :param skip_header (bool, default = True): Skip the header.
        :param dataset (str, default = 'train'): Dataset to load. Must exist as key in self.data_files.
        """
        fp = open(self.data_files[dataset])
        if skip_header:
            next(fp)

        data = []
        for line in self.reader(fp):
            data_line, datapoint = {}, Datapoint()  # TODO Look at moving all of this to the datapoint class.

            for field in self.train_fields:
                idx = field.index if self.ftype in ['CSV', 'TSV'] else field.cname
                data_line[field.name] = self.process_doc(line[idx].rstrip())

            for field in self.label_fields:
                idx = field.index if self.ftype in ['CSV', 'TSV'] else field.cname
                data_line[field.name] = line[idx].rstrip()

            for key, val in data_line.items():
                setattr(datapoint, key, val)
            data.append(datapoint)

        # TODO Think about this some more. Fine to run this when extending the dataset (but maybe not extend labels?)
        # TODO but not when loading dev or test because then the model will see the data before it's allowed to.
        # TODO Potentially just allow one class per file / dataset.
        if dataset == 'train':
            self.data = data
        elif dataset == 'dev':
            self.dev = data
        elif dataset == 'test':
            self.test = data

        fp.close()

    def load_labels(self, label_path, ftype: str = None, sep: str = None,
                    skip_header: bool = True, label_processor: t.Callable = None, label_ix: t.Union[int, str] = None):
        """Load labels.
        :param path (str): Path to data files.
        :param label_file (str): Filename of data file.
        :param ftype (str, default = 'CSV'): Filetype of the file.
        :param sep (str, optional): Separator to be used with T/CSV files.
        :param skip_header (bool): Skip the header.
        :param label_processor: Function to process labels.
        :param label_ix (int, str): Index or name of column containing labels.
        :return labels: Returns the loaded data.
        """
        path = label_path if label_path is not None else self.path
        ftype = ftype if ftype is not None else self.ftype
        sep = sep if sep is not None else self.sep

        labels = []
        fp = open(path)
        if skip_header:
            next(fp)

        for line in self.reader(fp, ftype, sep):
            if ftype in ['CSV', 'TSV']:
                labels.append(line[label_ix.rstrip()])

    def reader(self, fp, ftype: str = None, sep: str = None):
        """Instatiate the reader to be used.
        :param fp: Opened file.
        :param ftype (str, default = None): Filetype if loading external data.
        :param sep (str, default = None): Separator to be used.
        :return reader: Iterable object.
        """
        ftype = ftype if ftype is not None else self.ftype
        if ftype in ['CSV', 'TSV']:
            sep = sep if sep else self.sep
            reader = csv.reader(fp, delimiter = sep)
        else:
            reader = self.json_reader(fp)
        return reader

    def json_reader(self, fp: str):
        """Create a JSON reading object"""
        for line in fp:
            yield json.loads(line)

    def build_token_vocab(self, data: t.DataType):
        """Build vocab over dataset."""
        train_fields = self.train_fields
        self.token_counts = Counter()
        for doc in data:
            for f in train_fields:
                self.token_counts.update(getattr(doc, getattr(f, 'name')))
        self.token_counts.update({'<unk>': np.mean(list(self.token_counts.values()))})
        self.token_counts.update({'<pad>': 0})

        self.itos = {ix: tok for doc in data for ix, (tok, _) in enumerate(self.token_counts.most_common())}
        self.stoi = {tok: ix for ix, tok in self.itos.items()}

    def extend_vocab(self, data: t.DataType):
        """Extend the vocabulary."""
        for doc in data:
            start_ix = len(self.itos)
            for f in self.train_fields:
                tokens = getattr(doc, getattr(f, 'name'))
                self.token_counts.update(tokens)
                self.itos.update({start_ix + ix: tok for ix, tok in enumerate(tokens)})

        self.stoi = {tok: ix for ix, tok in self.itos.items()}

    def vocab_size(self):
        """Get the size of the vocabulary."""
        return len(self.itos)

    def vocab_token_lookup(self, tok: str):
        """Lookup a single token in the vocabulary.
        :param tok (str): Token to look up.
        :return ix: Return the index of the vocabulary item.
        """
        try:
            ix = self.stoi[tok]
        except IndexError as e:
            ix = self.stoi['<unk>']
        return ix

    def vocab_ix_lookup(self, ix: int):
        """Lookup a single index in the vocabulary.
        :param ix (int): Index to look up.
        :return tok: Returns token
        """
        return self.itos[ix]

    def build_label_vocab(self, labels):
        labels = set(getattr(l, getattr(f, 'name')) for l in labels for f in self.label_fields)
        self.itol = {ix: l for ix, l in enumerate(sorted(labels))}
        self.ltoi = {l: ix for ix, l in self.itol.items()}

    def label_name_lookup(self, label):
        """Look up label index from label."""
        return self.ltoi[label]

    def label_ix_lookup(self, label):
        """Look up label index from label."""
        return self.itol[label]

    def label_count(self):
        """Get the number of the labels."""
        return len(self.itol)

    def process_label(self, label, processor: t.Callable = None):
        """Modify label using external function to process it.
        :param label: Label to process.
        :param processor: Function to process the label."""
        processor = processor if processor is not None else self.label_processor
        return processor(label) if processor is not None else self.label_name_lookup[label]

    def process_doc(self, doc: t.DocType):
        if isinstance(doc, list):
            doc = " ".join(doc)

        if self.lower:
            doc = doc.lower()

        doc = self.tokenizer(doc.replace("\n", " "))

        if self.preprocessor is not None:
            doc = self.preprocessor(doc)

        if self.repr_transform is not None:
            doc = self.repr_transform(doc)

        if len(doc) > self.length:
            self.length = len(doc)

        return doc

    def pad(self, data: t.DataType, length: int = None):
        """Pad each document in the datasets in the dataset or trim document."""
        if not self.length and length is not None:
            self.length = length

        length = length if length is not None else self.length

        for doc in data:
            for field in self.train_fields:
                text = getattr(doc, getattr(field, 'name'))
                delta = length - len(text)
                yield text[:delta] if delta < 0 else ['<pad>'] * delta + text

    def onehot_encode(self, data):
        """Onehot encode a document."""
        self.encoded = [0] * len(data)
        for ix, doc in enumerate(data):
            encoded_doc = [0] * len(self.stoi)
            for f in self.train_fields:
                text = getattr(doc, getattr(f, 'name'))
                for tok in self.stoi:
                    encoded_doc[self.stoi[tok]] = 1 if tok in text else 0
                encoded_doc = torch.LongTensor(encoded_doc)
            setattr(doc, 'encoded', encoded_doc)
            self.encoded[ix] = encoded_doc
        return self.encoded

    def encode(self, data):
        self.encoded = [0] * len(data)
        for ix, doc in enumerate(data):
            for f in self.train_fields:
                text = getattr(doc, getattr(f, 'name'))
                encoded_doc = [0] * len(text)
                for i, w in enumerate(text):
                    try:
                        encoded_doc[i] = self.stoi[w]
                    except KeyError as e:
                        encoded_doc[i] = self.stoi['<unk>']
            setattr(doc, 'encoded', torch.LongTensor(encoded_doc))
            self.encoded[ix] = torch.LongTensor(encoded_doc)
        return self.encoded

    def stratify(self, data, strata_field):
        # TODO Rewrite this code to make sense with this implementation.
        # TODO This doesn't make sense to me.
        strata_maps = defaultdict(list)
        for doc in data:
            strata_maps[getattr(doc, strata_field)].append(doc)
        return list(strata_maps.values())

    def split(self, data: t.DataType, splits: t.Union[int, t.List[int]], stratify: str = None):
        """Split the dataset.
        :param data (t.DataType): Dataset to split.
        :param splits (int | t.List[int]]): Real valued splits.
        :param stratify (str): The field to stratify the data along.
        :return data: Return splitted data.
        """

        if stratify is not None:
            data = self.stratify(data, )

        if isinstance(splits, float):
            splits = [splits]

        num_splits = len(splits)
        num_datapoints = len(data)
        splits = list(map(lambda x: floor(num_datapoints * x), splits))

        for ix, split in enumerate(splits):
            if split == 0:
                if 1 < splits[ix - 1] and ix + 1 != len(splits):
                    splits[ix] = splits[ix - 1] + 1
                else:
                    splits[ix] = 1

        if num_splits == 1:
            return data[:splits[0]], data[splits[0]:]
        elif num_splits == 2:
            return data[:splits[0]], data[-splits[1]:]
        elif num_splits == 3:
            return data[:splits[0]], data[splits[0]:splits[1]], data[-splits[2]:]

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        try:
            return len(self.data)
        except TypeError:
            return 2**32

    def __iter__(self):
        for x in self.data:
            yield x

    def __getattr__(self, attr):
        if attr in self.fields_dict:
            for x in self.data:
                yield getattr(x, attr)
