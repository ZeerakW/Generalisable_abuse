import os
import re
import csv
import spacy
import torch
from collections import defaultdict
from torchtext import data
import gen.shared.types as t


class BatchGenerator:
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


# class FileReader(torch.utils.data.Dataset):
#
#     def __init__(self, dir: str, splits: t.Dict[str, str], sep: str = '\t', fields: t.List[str], shuffle: bool = True,
#                  skip_header: bool = True, **kwargs):
#         """Initialise data class.
#         :param dir (str): Directory containing dataset.
#         :param splits (str): t.Dictionary containing filenames type of data.
#         :param sep: Separator to be used.
#         :param fields: The data fields in the file.
#         :param shuffle (bool, default: True): Shuffle the data between each epoch.
#         :param skip_header (bool, default: True): Skip the first line.
#         """
#         self.dir = dir
#         self.splits = splits
#         self.ftype = ftype
#         self.sep = sep
#         self.fields = fields
#         self.shuffle = shuffle
#         self.skip_header = skip_header
#         self.tagger = spacy.load('en', disable = ['parser', 'tagger', 'ner', 'textcat'])
#         self.ftoi = {item: i for i, item in enumerate(fields)}
#
#     def read_files(one_hot: bool = True):
#         data = defaultdict(defaultdict(list))
#         for dataf, fh in self.splits.items():
#             if not fh:
#                 data[dataf] = None
#                 continue
#
#             with open(os.path.abspath(os.path.join(self.dir, fh)))) as fin:
#                 reader = csv.reader(fin, delimiter = self.sep)
#                 if self.skip_header:
#                     header = next(reader)
#
#                 for line in reader:
#                     for f, i in self.ftoi.items():
#                         data[datatype][f].append(line[i])
#
#         self.data = data['train'], data['dev'], data['test']
#
#
#     def _read_file(split, fh, t one_hot: bool = True):


class Dataset(data.TabularDataset):

    def __init__(self, data_dir: str, splits: t.Dict[str, str], ftype: str,
                 fields: t.List[t.Tuple[t.FieldType, ...]] = None, shuffle: bool = True,
                 skip_header: bool = True, repeat_in_batches = False, device: str = 'cpu'):
        """Initialise data class.
        :param data_dir (str): Directory containing dataset.
        :param fields (t.Dict[str, str]): The data fields in the file.
        :param splits (str): t.Dictionary containing filenames type of data.
        :param ftype: File type of the data file.
        :param shuffle (bool, default: True): Shuffle the data between each epoch.
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

    def init(self):
        super(Dataset, self).__init__(**self.load_params)

    @property
    def field_instances(self):
        """Set or return the instances of the fields used."""
        return self.field_types

    @field_instances.setter
    def field_instance(self, fields: t.Tuple[t.FieldType, ...]):
        self.field_types = fields

    @property
    def fields_obj(self):
        return self.dfields

    @fields_obj.setter
    def fields_obj(self, fields):
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
        if isinstance(field, t.List) or isinstance(attribute, t.List):
            if len(field) == 1 and len(attribute) > 1:
                for a, v in zip(attribute, val):
                    setattr(field[0], a, v)
            else:
                for f, a, v in zip(field, attribute, val):
                    setattr(f, a, v)
        else:
            setattr(field, attribute, val)
