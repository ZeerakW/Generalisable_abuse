import os
import re
import csv
import pdb
import spacy
import torch
from torchtext import data
import gen.shared.types as t
from torch.utils.data import IterableDataset, DataLoader


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


class GeneralDataset(IterableDataset):
    """A general dataset class, which loads a dataset, creates a vocabulary, pads, tensorizes, etc."""
    def __init__(self, data_dir: str, format: str, fields: t.List[t.Tuple[str, ...]],
                 batch_sizes: t.Union[int, t.Tuple[int]],
                 train: str, dev: str = None, test: str = None, train_labels: str = None, dev_labels: str = None,
                 test_labels: str = None, tokenizer: t.Union[t.Callable, str] = 'spacy', lower: bool = True,
                 preprocessor: t.Callable = None, transformations: t.Callable = None):
        """Initialize the variables required for the dataset loading.
        :param data_dir (str): Path of the directory containing the files.
        :param format (str): Format of the file ([C|T]SV and JSON accepted)
        :param fields (t.List[t.Tuple[str, ...]]): The names of the fields in the same order as they appear (in csv).
        :param batch_sizes (t.Union[int, t.Tuple[int]]): Single int or tuple of ints for batch sizes.
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
        """
        try:
            assert format.upper() in ['JSON', 'CSV', 'TSV']
            self.format = format
        except AssertionError as e:
            raise AssertionError("Input the correct file format: CSV/TSV or JSON")

        assert('label' in fields or train_labels)

        self.data_dir = os.path.abspath(data_dir)
        self.fields = fields
        self.fields_dict = dict(fields)
        self.batch_size = batch_sizes
        self.data_files = {key: os.path.join(self.data_dir, f) for f, key in zip([train, dev, test],
                                                                                 ['train', 'dev', 'test']) if f}
        self.label_files = {key: os.path.join(self.data_dir, f) for f, key in
                            zip([train_labels, dev_labels, test_labels], ['train', 'dev', 'test']) if f}

        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.data_dir = data_dir
        self.repr_transform = transformations

    def load(self, skip_header = True):
        """Load the dataset."""

        if self.format.upper() in ['CSV', 'TSV']:
            reader = csv.reader()
            # TODO
            # Setup reader
            # Skip line
            # Read line
            # Tokenize
            # Vectorize

        else:
            reader = self.json_reader()

        raise NotImplementedError

    def load_external(self, skip_header = True):
        """Load another dataset without influencing the vocabulary."""
        raise NotImplementedError

    def build_vocab(self, data: t.DataType = None):
        """Build vocab over dataset."""
        self.itos = {ix: tok for doc in data for ix, tok in enumerate(doc)}
        self.stoi = {tok: ix for ix, tok in self.itos.items()}

    def vocab_size(self):
        """Get the size of the vocabulary."""
        return len(self.itos)

    def vocab_lookup(self, tok: str):
        """Lookup a single token in the vocabulary.
        :param tok (str): Token to look up.
        :return ix: Return the index of the vocabulary item.
        """
        try:
            ix = self.stoi[tok]
        except IndexError as e:
            ix = self.stoi['<UNK>']

        return ix

    def index_lookup(self, ix: int):
        """Lookup a single index in the vocabulary.
        :param ix (int): Index to look up.
        :return tok: Returns token
        """
        return self.itos[ix]

    def pad(self, data: t.DataType, length: int = None):
        """Pad each document in the datasets in the dataset."""
        raise NotImplementedError

    def preprocess(self, doc: t.DocType):
        if isinstance(doc, list):
            doc = " ".join(doc)

        if self.lower:
            doc = doc.lower()

        doc = self.tokenizer(doc.replace("\n", " "))

        if self.preprocessor is not None:
            doc = self.preprocessor(doc)

        return doc

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        try:
            return len(self.examples)
        except TypeError:
            return 2**32

    def __iter__(self):
        for x in self.examples:
            yield x

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)

    def stratify(self, data, strata_field):
        # TODO Rewrite this code to make sense with this implementation.
        # Taken from torchtext.data
        unique_strata = set(getattr(doc, strata_field) for doc in data)
        strata_maps = {s: [] for s in unique_strata}

        for doc in data:
            strata_maps[getattr(doc, strata_field)].append(doc)
        return list(strata_maps.values())

    def split(self, splits: t.Union[int, t.List[int]], data: t.DataType = None):
        raise NotImplementedError


class Dataset(data.TabularDataset):

    def __init__(self, data_dir: str, splits: t.Dict[str, str], ftype: str,
                 fields: t.List[t.Tuple[t.FieldType, ...]] = None, shuffle: bool = True, sep: str = 'tab',
                 skip_header: bool = True, repeat_in_batches = False, device: str = 'cpu'):
        """Initialise data class.
        :param data_dir (str): Directory containing dataset.
        :param fields (t.Dict[str, str]): The data fields in the file.
        :param splits (str): t.Dictionary containing filenames type of data.
        :param ftype: File type of the data file.
        :param shuffle (bool, default: True): Shuffle the data between each epoch.
        :param sep (str): Seperator (if csv/tsv file).
        :param repeat_in_batches (bool, default: False): Repeat data within batches
        :param device (str, default: 'cpu'): Device to process on
        """
        self.tagger = spacy.load('en', disable = ['ner'])
        num_files = len(splits.keys())  # Get the number of datasets.

        self.load_params = {'path': data_dir,
                            'format': ftype,
                            'fields': fields,
                            'skip_header': skip_header,
                            'num_files': num_files}
        self.load_params.update(splits)
        self.dfields = fields
        self.repeat = repeat_in_batches
        self.device = device

    @property
    def field_instances(self):
        """Set or return the instances of the fields used."""
        return self.field_types

    @field_instances.setter
    def field_instance(self, fields: t.Tuple[t.FieldType, ...]):
        self.field_types = fields

    @property
    def fields(self):
        return self.dfields

    @fields.setter
    def fields(self, fields):
        self.dfields = fields
        self.load_params.update({'fields': fields})

    @property
    def data_params(self):
        return self.load_params

    @data_params.setter
    def data_params(self, params):
        self.load_params.update(params)

    def load_data(self) -> t.Tuple[t.DataType, ...]:
        """Load the dataset and return the data.
        :return data: Return loaded data.
        """
        if self.load_params['num_files'] == 1:
            train = self._data(**self.load_params)
            self.data = (train, None, None)
        elif self.load_params['num_files'] == 2:
            train, test = self._data(**self.load_params)
            self.data = (train, None, test)
        elif self.load_params['num_files'] == 3:
            train, dev, test = self._data(**self.load_params)
            self.data = (train, dev, test)
        return self.data

    @classmethod
    def _data(cls, path: str, format: str, fields: t.Union[t.List[t.Tuple[t.FieldType, ...]], t.Dict[str, tuple]],
              train: str, validation: str = None, test: str = None, skip_header: bool = True,
              num_files: int = 3) -> t.Tuple[t.DataType, ...]:
        """Use the loader in torchtext.
        :param path (str): Directory the data is stored in.
        :param format (str): Format of the data.
        :param fields (t.Union[t.List[t.FieldType], t.Dict[str tuple]]): Initialised fields.
        :param train (str): Filename of the training data.
        :param validation (str, default: None): Filename of the development data.
        :param test (str, default: None): Filename of the test data.
        :param skip_header (bool, default: True): Skip first line.
        :param num_files (int, default: 3): Number of files/datasets to load.
        :return data: Return loaded data.
        """
        splitted = data.TabularDataset.splits(path = path, format = format, fields = fields, train = train,
                                              validation = validation, test = test, skip_header = skip_header)
        return splitted

    def clean_document(self, text: t.DocType, processes: t.List[str] = None):
        """Data cleaning method.
        :param text (t.DocType): The document to be cleaned.
        :param processes (t.List[str]): The cleaning processes to be undertaken.
        :return cleaned: Return the cleaned text.
        """
        cleaned = str(text)
        if 'lower' in self.cleaners or 'lower' in processes:
            cleaned = cleaned.lower()
        if 'url' in self.cleaners or 'url' in processes:
            cleaned = re.sub(r'https?:/\/\S+', '<URL>', cleaned)
        if 'hashtag' in self.cleaners or 'hashtag' in processes:
            cleaned = re.sub(r'#[a-zA-Z0-9]*\b', '<HASHTAG>', cleaned)
        if 'username' in self.cleaners or 'username' in processes:
            cleaned = re.sub(r'@\S+', '<USER>', cleaned)

        return cleaned

    def tokenize(self, document: t.DocType, processes: t.List[str] = None):
        """Tokenize the document using SpaCy and clean it as it is processed.
        :param document: Document to be parsed.
        :param processes: The cleaning processes to engage in.
        :return toks: Document that has been passed through spacy's tagger.
        """
        if processes:
            toks = [tok.text for tok in self.tagger(self.clean_document(document, processes = processes))]
        else:
            toks = [tok.text for tok in self.tagger(self.clean_document(document))]
        return toks

    @property
    def data_attr(self):
        """Get or set the data attribute."""
        return self.data

    @data_attr.setter
    def data_attr(self, data: t.Tuple[t.DataType, ...]):
        if len(data) == 1:
            self.data = (data[0], None, None)
        elif len(data) == 2:
            self.data = (data[0], None, data[1])
        elif len(data) == 3:
            self.data = data

    def generate_batches(self, sort_func: t.Callable, datasets: t.Tuple[t.DataType, ...] = None,
                         batch_sizes: t.Tuple[int, ...] = (32,)):
        """Create the minibatching here.
        :param train (t.DataType, optional): Provide a processed train dataset.
        :param test (t.DataType, optional): Provide a processed test dataset.
        :param dev (t.DataType, optional): Provide a processed test dataset.
        :param batch_size (t.Tuple[int, ...]): Define batch sizes.
        :return ret: Return the batched data.
        """
        self.data_attr = datasets
        batches = data.BucketIterator.splits(self.data,
                                             batch_sizes,
                                             sort_key = sort_func,
                                             device = self.device,
                                             sort_within_batch = True, repeat = self.repeat)
        return batches

    def tag(self, document: t.DocType):
        """Tag document using spacy.
        :param document: Document to be parsed.
        :return doc: Document that has been passed through spacy's tagger.
        """
        doc = self.tagger(self.clean_document(document))
        return doc

    def get_spacy_annotations(self, document: t.DocType, processes: t.List[str]) -> t.Tuple:
        """Get annotations from SpaCy requested.
        :param document: The document to process.
        :param processes: The annotation processes to get.
        :return res (tuple): t.Tuple containing annotations requested.
        """
        res = [(tok.text, tok.pos_, tok.dep_, (tok.dep_, tok.head.dep_)) for tok in document]
        token, pos, dep, head = zip(*res)

        res = [None, None, None, None]

        if 'token' in processes:
            res[0] = token
        if 'pos' in processes:
            res[1] = pos
        if 'dep' in processes:
            res[2] = dep
        if 'head' in processes:
            res[3] = head
        if 'children' in processes:
            raise NotImplementedError

        return tuple(res)

    def set_field_attribute(self, field: t.Union[t.FieldType, t.List[t.FieldType]],
                            attribute: t.Union[str, t.List[str]],
                            val: t.Union[t.Any, t.List[t.Any]]):
        """Take an initialised field and an attribute.
        :param field (t.FieldType): The field to be modified.
        :param attribute (str): The attribute to modify.
        :param val (t.AllBuiltin): The new value of the attribute.
        """
        if isinstance(field, t.List):
            for f, a, v in zip(field, attribute, val):
                setattr(f, a, v)
        else:
            setattr(field, attribute, val)
