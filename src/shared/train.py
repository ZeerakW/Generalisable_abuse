from tqdm import tqdm
import src.shared.types as t
from src.shared.data import Dataset, BatchGenerator
from src.shared.clean import Cleaner


def read_liwc() -> dict:
    with open('~/Users/zeerakw/Documents/PhD/projects/Generalisable_abuse/data/liwc-2015.csv', 'r') as liwc_f:
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


def compute_unigram_liwc(doc: t.DocType):
    """Compute LIWC for each document document.
    :param doc (t.DocType): Document to operate on.
    :return liwc_doc (t.DocType): Document represented as LIWC categories.
    """
    liwc_doc = []
    kleene_star = [k[:-1] for k in liwc_dict if k[-1] == '*']

    for w in doc:
        if w in liwc_dict:
            liwc_doc.extend(liwc_dict[w])
        else:
            # This because re.findall is slow.
            candidates = [r for r in kleene_star if r in w]  # Find all candidates

            num_cands = len(candidates)
            if num_cands == 0:
                term = 'UNK'
                continue
            elif num_cands == 1:
                term = candidates[0]
            elif num_cands > 1:
                sorted_cands = sorted(candidates, key=len, reverse = True)
                term = sorted_cands[0] + '*'
            liwc_doc.append(term)

    assert(len(liwc_doc) == len(doc))

    return liwc_doc


def store_fields(obj, data_field, label_field, **kwargs):
    """Store fields in the dataset object. Final two fields always label, train.
    :param data (t.FieldType): The field instance for the data.
    :param label (t.FieldType): The field instance for the label.
    :param kwargs: Will search for any fields in this.
    """
    if kwargs:
        field = []
        for k in kwargs:
            if 'field' in k:
                field.append(kwargs[k])
    field.extend([label_field, data_field])
    obj.field_instance = tuple(field)


def create_batches(data_dir: str, splits: t.Dict[str, t.Union[str, None]], ftype: str, fields: t.Union[dict, list],
                   cleaners: t.List[str], batch_sizes: t.Tuple[int, ...], shuffle: bool, sep: str, skip_header: bool,
                   repeat_in_batches: bool, device: t.Union[str, int],
                   data_field: t.Tuple[t.FieldType, t.Union[t.Dict, None]],
                   label_field: t.Tuple[t.FieldType, t.Union[t.Dict, None]], **kwargs):

    # Initiate the dataset object
    data = Dataset(data_dir = data_dir, splits = splits, ftype = ftype, fields = fields, cleaners = cleaners,
                   shuffle = shuffle, sep = sep, skip_header = skip_header, repeat_in_batches = repeat_in_batches,
                   device = device)

    # If the fields need new attributes set: set them.
    # TODO assumes only data and field labels need modification.
    if data_field[1]:
        data.set_field_attribute(data_field[0], data_field[1]['attribute'], data_field[1]['value'])

    if label_field[1]:
        data.set_field_attribute(label_field[0], label_field[1]['attribute'], label_field[1]['value'])

    # Store our Field instances so we can later access them.
    store_fields(data, data_field, label_field, **kwargs)

    data.fields = fields  # Update the fields in the class

    loaded = data.load_data()  # Data paths exist in the class

    if len([v for v in splits.values() if v is not None]) == 1:  # If only one dataset is given
        train, test = data.split(split_ratio = kwargs['split_ratio'], stratified = True, strata_field = kwargs['label'])
        loaded = (train, None, test)

    data_field.build_vocab()
    label_field.build_vocab()

    train, dev, test = data.generate_batches(lambda x: len(x.data), loaded, batch_sizes)
    train_batch = BatchGenerator(train, 'data', 'label')
    dev_batch = None if dev is None else BatchGenerator(dev, 'data', 'label')
    test_batch = BatchGenerator(test, 'data', 'label')

    batches = (train_batch, dev_batch, test_batch)
    return data, batches


def setup_data():
    device = 'cpu'
    data_dir = '/Users/zeerakw/Documents/PhD/projects/Generalisable_abuse/data/'
    clean = Cleaner()

    # MFTC
    text = (t.text_data, {'attribute': ['tokenize', 'preprocess'],
                          'value': [clean.tokenize, compute_unigram_liwc]})
    label = (t.int_label, None)

    fields = [('CF_count', None), ('hate_speech', None), ('offensive', None), ('neither', None), ('label', label[0]),
              ('data', text)]

    data_opts = {'splits': {'train': 'davidson_offensive'}, 'ftype': 'csv', 'data_field': text, 'fields': fields,
                 'label_field': label, 'batch_sizes': (64,), 'shuffle': True, 'sep': ',', 'skip_header': True,
                 'repeat_in_batches': False}

    ds = create_batches(data_dir = data_dir, device = device, **data_opts)

    return ds


def train(epochs, model, loss_func, optimizer):

    ds, train, dev, test = setup_data()

    for epoch in tqdm(range(epochs)):
        model.zero_grad()
        batch = next(iter(train))

        scores = model(batch)
        loss = loss_func(scores, batch.labels)
        loss.backward()
        optimizer.step()
