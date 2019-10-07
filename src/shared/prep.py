import os
import json
import torchtext
import spacy
from collections import defaultdict
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

        [splits.update({k: data_dir + '/' + splits[k]}) for k in splits.keys()]

        num_files = len(splits.keys())  # Get the number of datasets.
        if num_files == 1:
            train = super(Dataset, self).splits(path = data_dir, format = ftype, fields = fields,
                                                skip_header = skip_header, **splits)
        elif num_files == 2:
            train, test = super(Dataset, self).splits(path = data_dir, format = ftype, fields = fields,
                                                      skip_header = skip_header, **splits)
        elif num_files == 3:
            train, dev, test = super(Dataset, self).splits(path = data_dir, format = ftype, fields = fields,
                                                           skip_header = skip_header, **splits)




        self.batch_size = batch_size
        self.tagger = spacy.load('en', disable = ['ner'])

    def load_data(self, splits: Dict[str, str], fields: List[str]) -> tuple:
        """Load the dataset and any splits.
        :param splits (Dict[str, str]): Dictionary containing dataset splits and file names.
        :param fields: The fields of the data that we are interested in. The final entry is always the label.
        :return ret_val (tuple): 3-tuple of train, dev, and test data.
        """
        data_dict, label_dict = {}, {}
        for k, v in splits:
            data_dict[k], label_dict[k] = self._load_data(os.path.abspath(self.data_dir + '/' + v), fields)

        ret_val = []

        if 'train' in data_dict.keys():
            ret_val[0] = data_dict['train']
            self.train = data_dict['train']
            self.trainY = label_dict['train']

        if 'dev' in data_dict.keys():
            ret_val[1] = data_dict['dev']
            self.devY = label_dict['dev']
            self.dev = data_dict['dev']
        else:
            ret_val[1] = None

        if 'test' in data_dict.keys():
            ret_val[2] = data_dict['test']
            self.testY = label_dict['test']
            self.test = data_dict['test']
        else:
            ret_val[2] = None

        return ret_val

    @classmethod
    def _load_data(cls, fp: str,
                   fields: Union[Dict[str, int], Dict[str, str]],
                   label: Union[str, int], skip_header = False) -> Tuple[types.NPData, types.NPData, ...]:
        """The actual data loading method.
        :param fp: Full path to the file.
        :param fields (Dict[str, int]): The name of the fields and potentially the position in the csv.
        :param label: Label index or key.
        :param skip_header (bool, optional. Default: False): Skip the header in the file.
        :return ...: The loaded dataset and the labels.
        """
        output_dict = defaultdict(list)
        labels = []

        with open(fp, 'r', encoding = 'utf-8') as fin:
            for line in fin:
                if cls.ftype in ['csv', 'tsv']:
                    loaded = line.split(cls.sep)
                    for k in fields.keys():
                        output_dict[k].append(loaded[fields[k]])  # TODO Double check this.
                elif cls.ftype in ['json']:
                    loaded = json.loads(line)
                    for k in output_dict.keys():
                        output_dict[k].append(loaded[k])  # TODO Double check this.

                labels.append(fields[label])
        return output_dict, labels

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

    def extract_dep_neighbours(self, document: types.DocType) -> List[Tuple[str, str, ...]]:
        """Extract the dependency neighbours of each word.
        :param document (types.DocType, required): Spacy tagged document.
        :return tags (List[Tuple[str]]): Neighbours in the dependency tree of the current token.
        """
        raise NotImplementedError
