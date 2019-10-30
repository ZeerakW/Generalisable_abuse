import re
import pdb
import gen.shared.types as t
from gen.shared.data import Dataset, BatchGenerator


def read_liwc() -> dict:
    with open('/Users/zeerakw/Documents/PhD/projects/Generalisable_abuse/data/liwc-2015.csv', 'r') as liwc_f:
        liwc_dict = {}
        for line in liwc_f:
            k, v = line.strip('\n').split(',')
            if k in liwc_dict:
                liwc_dict[k] += [v]
            else:
                liwc_dict.update({k: [v]})

    return liwc_dict


global liwc_dict
liwc_dict = read_liwc()

# TODO Different ways of representing LIWC
# First category
# Take counts of categories, use the heighest count
# Basically look at the if statement and see what comes out there.


def compute_unigram_liwc(doc: t.DocType):
    """Compute LIWC for each document document.
    :param doc (t.DocType): Document to operate on.
    :return liwc_doc (t.DocType): Document represented as LIWC categories.
    """
    liwc_doc = []
    kleene_star = [k[:-1] for k in liwc_dict if k[-1] == '*']

    for w in doc:
        if w in liwc_dict:
            liwc_doc.append(liwc_dict[w][0])
        else:
            # This because re.findall is slow.
            candidates = [r for r in kleene_star if r in w]  # Find all candidates

            num_cands = len(candidates)
            if num_cands == 0:
                term = 'NUM' if re.findall(r'[0-9]+', w) else 'UNK'
            elif num_cands == 1:
                term = candidates[0]
            elif num_cands > 1:
                sorted_cands = sorted(candidates, key=len, reverse = True)
                term = sorted_cands[0] + '*'
            liwc_doc.append(term)
    try:
        assert(len(liwc_doc) == len(doc))
    except AssertionError:
        pdb.set_trace()

    return " ".join(liwc_doc)


def store_fields(obj, data_field, label_field, **kwargs):
    """Store fields in the dataset object. Final two fields always label, train.
    :param data (t.FieldType): The field instance for the data.
    :param label (t.FieldType): The field instance for the label.
    :param kwargs: Will search for any fields in this.
    """
    field = []
    if kwargs:
        for k in kwargs:
            if 'field' in k:
                field.append(kwargs[k])
    field.extend([label_field, data_field])
    obj.field_instance = tuple(field)


def create_batches(data_dir: str, splits: t.Dict[str, t.Union[str, None]], ftype: str, fields: t.Union[dict, list],
                   batch_sizes: t.Tuple[int, ...], shuffle: bool, skip_header: bool, repeat_in_batches: bool,
                   device: t.Union[str, int], data_field: t.Tuple[t.FieldType, t.Union[t.Dict, None]],
                   label_field: t.Tuple[t.FieldType, t.Union[t.Dict, None]], **kwargs):

    # Initiate the dataset object
    data = Dataset(data_dir = data_dir, splits = splits, ftype = ftype, fields = fields,
                   shuffle = shuffle, skip_header = skip_header, repeat_in_batches = repeat_in_batches,
                   device = device)

    # If the fields need new attributes set: set them.
    # TODO assumes only data and field labels need modification.
    if data_field[1]:
        data.set_field_attribute([data_field[0]], data_field[1]['attribute'], data_field[1]['value'])

    if label_field[1]:
        data.set_field_attribute([label_field[0]], label_field[1]['attribute'], label_field[1]['value'])

    # Store our Field instances so we can later access them.
    store_fields(data, data_field, label_field, **kwargs)

    data.fields_obj = fields  # Update the fields in the class

    pdb.set_trace()
    loaded = data.load_data()  # Data paths exist in the class

    if len([v for v in splits.values() if v is not None]) == 1:  # If only one dataset is given
        strata_label = kwargs['label'] if 'label' in kwargs else 'label'
        pdb.set_trace()
        train, test = data.split(split_ratio = kwargs['split_ratio'], stratified = True, strata_field = strata_label)
        loaded = (train, None, test)

    data_field.build_vocab()
    label_field.build_vocab()

    train, dev, test = data.generate_batches(lambda x: len(x.data), loaded, batch_sizes)
    train_batch = BatchGenerator(train, 'data', 'label')
    dev_batch = None if dev is None else BatchGenerator(dev, 'data', 'label')
    test_batch = BatchGenerator(test, 'data', 'label')

    batches = (train_batch, dev_batch, test_batch)
    return data, batches, data_field.vocab
