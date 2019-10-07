import re
import torchtext
import spacy
from typing import List, Tuple, Dict, Union
import src.shared.types as types


class Dataset(torchtext.data.TabularDataset):

    def __init__(self, data_dir: str, fields: types.FieldType, splits: Dict[str, str], ftype: str = 'tsv',
                 batch_size: int = 64, shuffle: bool = True, sep: str = 'tab', skip_header: bool = True):
        """Initialise data class.
        :param data_dir (str): Directory containing dataset.
        :param fields (Dict[str, str]): The data fields in the file.
        :param splits (str): Dictionary containing filenames type of data.
        :param ftype: File type of the data file.
        :param batch_size (int):
        :param shuffle: Shuffle the data between each epoch.
        :param sep: Seperator (if csv/tsv file).
        """

        self.tagger = spacy.load('en', disable = ['ner'])
        [splits.update({k: data_dir + '/' + splits[k]}) for k in splits.keys()]
        num_files = len(splits.keys())  # Get the number of datasets.

        data_load = {'path': data_dir,
                     'format': ftype,
                     'fields': fields,
                     'skip_header': skip_header,
                     'num_files': num_files}
        data_load.update(splits)

        if num_files == 1:
            train = self.load_data(data_load)
            self.data = (train, None, None)
        elif num_files == 2:
            train, test = self.load_data(data_load)
            self.data = (train, None, test)
        elif num_files == 3:
            train, dev, test = self.load_data(data_load)
            self.data = (train, dev, test)

        self.batch_size = batch_size

    def clean_document(self, text: types.DocType, processes: List[str]):
        """Data cleaning method.
        :param text (types.DocType): The document to be cleaned.
        :param processes (List[str]): Strings of the cleaning processes to take.
        :return cleaned: Return the cleaned text.
        """
        cleaned = str(text)
        if 'lower' in processes:
            cleaned = cleaned.lower()
        if 'url' in processes:
            cleaned = re.sub(r'https?:/\/\S+', '<URL>', cleaned)
        if 'hashtag' in processes:
            cleaned = re.sub(r'#[a-zA-Z0-9]*\b', '<HASHTAG>', cleaned)

        return cleaned

    @classmethod
    def load_data(cls, path: str, format: str, fields: Union[List[Tuple[types.FieldType, ...]], Dict[str, tuple]],
                  train: str, validation: str = None, test: str = None, skip_header: bool = True,
                  num_files: int = 3) -> Tuple[types.NPData, ...]:
        """Load the dataset and return the data.
        :param path (str): Directory the data is stored in.
        :param format (str): Format of the data.
        :param fields (types.FieldType): Initialised fields.
        :param train (str): Filename of the training data.
        :param validation (str, default: None): Filename of the development data.
        :param test (str, default: None): Filename of the test data.
        :param skip_header (bool, default: True): Skip first line.
        :return data: Return loaded data.
        """
        data = super(Dataset, cls).splits(path = path, format = format, fields = fields, train = train,
                                          validation = validation, test = test, skip_header = skip_header)
        return data

    def ix_to_label(self, label_to_ix):
        """Take label-index mapping and create map index to label.
        :param label_to_ix: Dictionary containing label_to_index mapping.
        :return ix_to_label: Dictionary containing index_to_label mapping.
        """
        ix_to_label = {v: k for k, v in label_to_ix.items()}
        return ix_to_label

    def label_to_ix(self, labels):
        """Generate labels to generate indice dictionary.
        :param labels: list of all labels in the dataset.
        :return label_to_ix: Dictionary to convert labels to indices.
        """
        if not isinstance(labels, set):
            labels = set(labels)

        label_to_ix = {}
        for label in labels:
            label_to_ix[label] = len(label_to_ix)

        return label_to_ix

    def tag(self, document: types.DocType):
        """Tag document using spacy.
        :param document: Document to be parsed.
        :return doc: Document that has been passed through spacy's tagger.
        """
        doc = self.tagger(document)
        return doc

    def extract_tokens(self, document: types.DocType) -> List[str]:
        """Extract tokens from the document.
        :param document: Spacy tagged document.
        :return tokens: List of strings containing tokens.
        """
        tokens = [tok.text for tok in document]
        return tokens

    def extract_pos(self, document: types.DocType) -> List[str]:
        """Extract POS from the document.
        :param document: Spacy tagged document.
        :return tags: List of strings containing POS tags.
        """
        tags = [tok.pos_ for tok in document]
        return tags

    def extract_dep(self, document: types.DocType) -> List[str]:
        """Extract dependency tags from the document.
        :param document: Spacy tagged document.
        :return tags: List of strings containing dependency tags.
        """
        tags = [tok.dep_ for tok in document]
        return tags

    def extract_head_dep(self, document: types.DocType) -> List[Tuple[str, str]]:
        """Extract head of words and the dependency of the current word from the document.
        :param document: Spacy tagged document.
        :return tags: List of tuple of strings containing dependency tag of words and their heads.
        """
        tags = [(tok.dep_, tok.head.dep_) for tok in document]
        return tags

    def extract_dep_children(self, document: types.DocType) -> List[Tuple[str, str, ...]]:
        """Extract the dependency neighbours of each word.
        :param document (types.DocType, required): Spacy tagged document.
        :return tags (List[Tuple[str]]): Neighbours in the dependency tree of the current token.
        """
        raise NotImplementedError
