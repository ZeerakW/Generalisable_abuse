import gen.shared.custom_types as t
import pdb
from gen.shared.data import Dataset, BatchGenerator


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
