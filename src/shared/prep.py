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
