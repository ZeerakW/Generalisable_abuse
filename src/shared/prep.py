import spacy
from typing import List
import src.shared.types as types


class Dataset(object):

    def __init__(self, data_dir: str, ftype: str, sep: str = 'tab'):
        """Initialise data class.
        :param data_dir: Path to dataset.
        :param ftype: File type of the data file.
        :param sep: Seperator (if csv/tsv file).
        """
        self.data_dir = data_dir
        self.ftype = ftype
        self.sep = sep
        self.tagger = spacy.load('en')

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
