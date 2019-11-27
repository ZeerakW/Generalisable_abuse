import os
import re
import csv
import json
import spacy
import torch
from torchtext import data
import gen.shared.types as t
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


class Field(object):
    """A class to set different properties of the individual fields."""
    def __init__(self, name: str, train: bool, label: bool, ignore: bool = True, ix: int = None, cname: str = None):
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
                 batch_sizes: t.Union[int, t.Tuple[int]],
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
        :param label_processor(t.Callable, default = None): Function to process labels with.
        """
        self.data_dir = os.path.abspath(data_dir)
        try:
            ftype = ftype.upper()
            assert ftype in ['JSON', 'CSV', 'TSV']
            self.ftype = ftype
        except AssertionError as e:
            raise AssertionError("Input the correct file ftype: CSV/TSV or JSON")

        assert('label' in fields or train_labels)

        self.sep = sep
        self.fields = fields
        self.fields_dict = dict(fields)
        self.batch_size = batch_sizes
        self.data_files = {key: os.path.join(self.data_dir, f) for f, key in zip([train, dev, test],
                                                                                 ['train', 'dev', 'test'])
                                                                                 if f is not None}
        self.label_files = {key: os.path.join(self.data_dir, f) for f, key in
                            zip([train_labels, dev_labels, test_labels], ['train', 'dev', 'test']) if f is not None}

        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.data_dir = data_dir
        self.repr_transform = transformations
        self.label_processor = label_processor if label_processor else self.label_processing
        self.length = 0

    def load(self, dataset: str = 'train', skip_header = True):
        """Load the dataset.
        :param skip_header (bool, default = True): Skip the header.
        :param dataset (str, default = 'train'): Dataset to load. Must exist as key in self.data_files.
        """
        fp = open(self.data_files[dataset])
        if self.skip_header or skip_header:
            next(fp)

        with self.reader(fp) as reader:
            self.data = []
            for line in reader:
                datapoint = {}  # TODO Look at moving all of this to the datapoint class.
                dp = Datapoint()
                for field in self.fields:
                    if not any([field.ignore, field.label]):
                        if self.ftype in ['CSV', 'TSV']:
                            datapoint[field.name] = self.preprocess(line[field.ix].rstrip())
                        else:
                            datapoint[field.name] = self.preprocess(line[field.cname].rstrip())

                    elif field.label:
                        if self.ftype in ['CSV', 'TSV']:
                            datapoint[field.name] = self.process_label(line[field.ix].rstrip())
                        else:
                            datapoint[field.name] = self.process_label(line[field.cname].rstrip())

                for key, val in datapoint.items():
                    setattr(dp, key, val)
                self.data.append(dp)

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
            reader = csv.reader(fp, sep = sep)
        else:
            reader = self.json_reader(fp)
        return reader

    def json_reader(self, fp: str):
        """Create a JSON reading object"""
        for line in fp:
            yield json.loads(line)

    def build_vocab(self, data: t.DataType = None):
        """Build vocab over dataset."""
        self.token_counts = Counter([tok for doc in data for tok in doc])
        self.itos = {ix: tok for doc in data for ix, (tok, _) in enumerate(self.token_counts.most_common())}
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

    def ix_lookup(self, ix: int):
        """Lookup a single index in the vocabulary.
        :param ix (int): Index to look up.
        :return tok: Returns token
        """
        return self.itos[ix]

    def build_label_vocab(self, labels):
        self.itol = {ix: l for ix, l in enumerate(labels)}
        self.ltoi = {l: ix for ix, l in self.itol.items()}

    def label_lookup(self, label):
        """Look up label index from label."""
        return self.ltoi[label]

    def label_ix_lookup(self, label):
        """Look up label index from label."""
        return self.ltoi[label]

    def process_label(self, label, processor: t.Callable = None):
        """Modify label using external function to process it.
        :param label: Label to process.
        :param processor: Function to process the label."""
        processor = processor if processor is not None else self.label_processor
        return processor(label) if processor is not None else self.label_lookup[label]

    def preprocess(self, doc: t.DocType):
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
            delta = len(doc) - length
            if delta < 0:
                yield doc[:delta]
            elif delta > 0:
                yield ['<pad>'] * delta + doc

    def onehot_encode(self, data):
        """Onehot encode a document."""
        self.encoded = []
        for doc in data:
            self.encoded.append([1 if tok in doc else 0 for tok in self.stoi])
        return self.encoded

    def encode(self, data):
        self.encoded = []
        for doc in data:
            encode_doc = []
            for w in doc:
                try:
                    ix = self.stoi[w]
                except IndexError as e:
                    ix = self.stoi['<unk>']
                encode_doc.append(ix)
            self.encoded.append(encode_doc)
        return self.encoded

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

    def stratify(self, data, strata_field):
        # TODO Rewrite this code to make sense with this implementation.
        # Taken from torchtext.data
        # unique_strata = set(getattr(doc, strata_field) for doc in data)
        strata_maps = defaultdict(list)

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
