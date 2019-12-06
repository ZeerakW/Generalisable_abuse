import unittest
from gen.shared.data import Field, GeneralDataset, Datapoint
import pdb


class TestDataSet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up class with data as well."""
        fields = [Field('text', train = True, label = False, ignore = False, ix = 0, cname = 'text'),
                  Field('label', train = False, label = True, cname = 'label', ignore = False, ix = 1)]

        cls.csv_dataset = GeneralDataset(data_dir = '~/PhD/projects/active/Generalisable_abuse/gen/shared/tests/',
                                         ftype = 'csv', fields = fields, train = 'train.csv', dev = None,
                                         test = 'test.csv', train_labels = None, tokenizer = lambda x: x.split(),
                                         lower = True, preprocessor = None, transformations = None,
                                         label_processor = None, sep = ',')
        cls.csv_dataset.load('train')
        cls.train = cls.csv_dataset.data
        cls.json_dataset = GeneralDataset(data_dir = '~/PhD/projects/active/Generalisable_abuse/gen/shared/tests/',
                                          ftype = 'json', fields = fields, train = 'train.json', dev = None,
                                          test = 'test.json', train_labels = None, tokenizer = lambda x: x.split(),
                                          lower = True, preprocessor = None, transformations = None,
                                          label_processor = None, sep = ',')

    @classmethod
    def tearDownClass(cls):
        """Take down class."""
        cls.csv_dataset = 0
        cls.json_dataset = 0

    def test_load(self):
        """Test dataset loading."""
        expected = [("me gusta comer en la cafeteria".lower().split(), "SPANISH"),
                    ("Give it to me".lower().split(), "ENGLISH"),
                    ("No creo que sea una buena idea".lower().split(), "SPANISH"),
                    ("No it is not a good idea to get lost at sea".lower().split(), "ENGLISH")]
        csv_train = self.train
        output = [(doc.text, doc.label) for doc in csv_train]
        self.assertListEqual(output, expected, msg = 'Data Loading failed.')

        self.json_dataset.load('train', skip_header = False)
        json_train = self.json_dataset.data
        output = [(doc.text, doc.label) for doc in json_train]
        self.assertListEqual(output, expected, msg = 'Data Loading failed.')
        self.assertIsInstance(json_train[0], Datapoint, msg = 'Data Loading failed gave wrong type.')

    def test_build_token_vocab(self):
        """Test vocab building method."""
        expected = set(['<pad>', '<unk>'] + list(sorted("""me gusta comer en la cafeteria Give it to me
                   No creo que sea una buena idea No it is not a good idea to get lost at sea""".lower().split())))
        self.csv_dataset.build_token_vocab(self.train)
        output = set(sorted(self.csv_dataset.stoi.keys()))
        self.assertSetEqual(output, expected, msg = 'Vocab building failed.')

    def test_extend_vocab(self):
        """Test extending vocab."""
        train = """<pad> <unk> me gusta comer en la cafeteria Give it to me
                No creo que sea una buena idea No it is not a good idea to get lost at sea""".lower().split()
        test = "Yo creo que si it is lost on me".lower().split()
        expected = set(train + test)
        self.csv_dataset.load('test')
        test = self.csv_dataset.test
        self.csv_dataset.build_token_vocab(self.train)
        self.csv_dataset.extend_vocab(test)
        output = list(self.csv_dataset.stoi.keys())
        self.assertListEqual(sorted(output), sorted(expected), msg = 'Vocab Extension Failed.')

    def test_load_test_from_different_file(self):
        """Test loading a secondary dataset (test/dev set) from a different file."""
        self.csv_dataset.load('test')
        test = self.csv_dataset.test
        expected  = ["Yo creo que si".lower().split(), "it is lost on me".lower().split()]
        self.assertListEqual(test[0].text, expected[0])
        self.assertListEqual(test[1].text, expected[1])

    def test_vocab_token_lookup(self):
        '''Test looking up in vocab.'''
        self.csv_dataset.build_label_vocab(self.train)
        expected = 0
        output = self.csv_dataset.vocab_token_lookup('me')
        self.assertEqual(output, expected, msg = 'Vocab token lookup failed.')

    def test_vocab_ix_lookup(self):
        '''Test looking up in vocab.'''
        self.csv_dataset.build_label_vocab(self.train)
        expected = 'me'
        output = self.csv_dataset.vocab_ix_lookup(0)
        self.assertEqual(output, expected, msg = 'Vocab ix lookup failed.')

    def test_vocab_size(self):
        """Test vocab size is expected size."""
        self.csv_dataset.build_token_vocab(self.train)
        output = self.csv_dataset.vocab_size()
        expected = 25
        self.assertEqual(output, expected, msg = 'Building vocab failed.')

    def test_build_label_vocab(self):
        """Test building label vocab."""
        self.csv_dataset.build_label_vocab(self.train)
        output = list(sorted(self.csv_dataset.ltoi.keys()))
        expected = ['ENGLISH', 'SPANISH']
        self.assertListEqual(output, expected, msg = 'Building label vocab failed.')

    def test_label_name_lookup(self):
        """Test looking up in label."""
        self.csv_dataset.build_label_vocab(self.train)
        output = self.csv_dataset.label_name_lookup('SPANISH')
        expected = 1
        self.assertEqual(output, expected, msg = 'label name lookup failed.')

    def test_label_ix_lookup(self):
        '''Test looking up in label.'''
        self.csv_dataset.build_label_vocab(self.train)
        output = self.csv_dataset.label_ix_lookup(1)
        expected = 'SPANISH'
        self.assertEqual(output, expected, msg = 'label ix lookup failed.')

    def test_label_count(self):
        """Test label size is expected."""
        self.csv_dataset.build_label_vocab(self.train)
        expected = self.csv_dataset.label_count()
        output = 2
        self.assertEqual(output, expected, msg = 'Test that label count matches labels failed.')

    def test_process_label(self):
        """Test label processing."""
        self.csv_dataset.build_label_vocab(self.train)
        expected = 1
        output = self.csv_dataset.process_label('SPANISH')
        self.assertEqual(output, expected, msg = 'Labelprocessor failed without custom processor')

        def processor(label):
            labels = {'SPANISH': 1, 'ENGLISH': 0}
            return labels[label]

        expected = 1
        output = self.csv_dataset.process_label('SPANISH', processor = processor)
        self.assertEqual(output, expected, msg = 'Labelprocessor failed with custom processor')

    def test_no_preprocessing(self):
        """Test document processing."""
        setattr(self.csv_dataset, 'lower', False)
        setattr(self.csv_dataset, 'preprocessor', None)
        setattr(self.csv_dataset, 'repr_transform', None)

        inputs = "Give it to me, baby. Uhuh! Uhuh!"
        expected = inputs.split()
        output = self.csv_dataset.process_doc(inputs)
        self.assertListEqual(output, expected, msg = 'Process Document failed.')
        self.assertIsInstance(output, list, msg = 'Process document returned wrong type.')

    def test_lower_doc(self):
        """Test lowercasing processing."""
        setattr(self.csv_dataset, 'lower', True)
        setattr(self.csv_dataset, 'preprocessor', None)
        setattr(self.csv_dataset, 'repr_transform', None)

        inputs = "Give it to me, baby. Uhuh! Uhuh!"
        expected = inputs.lower().split()
        output = self.csv_dataset.process_doc(inputs)
        self.assertListEqual(output, expected, msg = 'Process Document failed with lowercasing.')
        self.assertIsInstance(output, list, msg = 'Process document with lowercasing produces wrong type.')

    def test_list_process_doc(self):
        """Test lowercasing processing."""
        setattr(self.csv_dataset, 'lower', True)
        setattr(self.csv_dataset, 'preprocessor', None)
        setattr(self.csv_dataset, 'repr_transform', None)

        inputs = "Give it to me, baby. Uhuh! Uhuh!"
        expected = inputs.lower().split()
        output = self.csv_dataset.process_doc(inputs.split())
        self.assertListEqual(output, expected, msg = 'Process Document failed with input type list.')
        self.assertIsInstance(output, list, msg = 'Process document with input type list produces wrong type.')

    def test_custom_preprocessor(self):
        """Test using a custom processor."""

        inputs = "Give it to me, baby. Uhuh! Uhuh!"
        expected = ["TEST" if '!' in tok else tok for tok in inputs.lower().split()]

        def preprocessor(doc):
            return ["TEST" if '!' in tok else tok for tok in doc]

        setattr(self.csv_dataset, 'lower', True)
        setattr(self.csv_dataset, 'preprocessor', preprocessor)
        output = self.csv_dataset.process_doc(inputs)
        self.assertListEqual(output, expected, msg = 'Process Document failed with preprocessor')
        self.assertIsInstance(output, list, msg = 'Process Document with preprocessor returned wrong type.')

    def test_repr_transformation(self):
        """Test using transformation to different representation."""

        inputs = "Give it to me, baby. Uhuh! Uhuh!"
        expected = "VERB DET PREP PRON NOUN AGREEMENT AGREEMENT".split()

        def transform(doc):
            transform = {'give': 'VERB', 'it': 'DET', 'to': "PREP", 'me': "PRON", 'baby': 'NOUN', 'uhuh!': 'AGREEMENT',
                         'uhuh': 'AGREEMENT'}
            return [transform[w.lower().replace(',', '').replace('.', '')] for w in doc]

        setattr(self.csv_dataset, 'repr_transform', transform)
        output = self.csv_dataset.process_doc(inputs)
        self.assertListEqual(output, expected, msg = 'Process Document failed with represntation transformation.')
        self.assertIsInstance(output, list, msg = 'Process Document with representation transformation returned wrong')

    def test_pad(self):
        """Test padding of document."""
        expected = [4 * ['<pad>'] + "me gusta comer en la cafeteria".split()]
        expected.append(6 * ['<pad>'] + ['give', 'it', 'to', 'me'])
        expected.append(3 * ['<pad>'] + ['no', 'creo', 'que', 'sea', 'una', 'buena', 'idea'])
        expected.append(0 * ['<pad>'] + ['no', 'it', 'is', 'not', 'a', 'good', 'idea', 'to', 'get', 'lost'])
        output = list(self.csv_dataset.pad(self.train, length = 10))
        self.assertListEqual(output, expected, msg = 'Padding doc failed.')

    def test_trim(self):
        """Test that trimming of the document works."""
        expected = 0 * ['<pad>'] + ['no', 'it', 'is', 'not', 'a', 'good', 'idea', 'to', 'get', 'lost'][:5]

        output = list(self.csv_dataset.pad(self.train, length = 5))[-1]
        self.assertListEqual(output, expected, msg = 'Zero padding failed.')

    def test_onehot_encoding(self):
        """Test the onehot encoding."""
        self.csv_dataset.build_token_vocab(self.train)
        self.csv_dataset.load('test')
        test = self.csv_dataset.test
        expected = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
        output = self.csv_dataset.onehot_encode(test)
        self.assertListEqual(output, expected, msg = 'Onehot encoding failed.')

    def test_encoding(self):
        """Test the encoding."""
        self.csv_dataset.build_label_vocab(self.train)
        self.csv_dataset.load('test')
        test = self.csv_dataset.test
        expected = [[6, 13, 14, 6], [1, 17, 22, 6, 0]]
        output = self.csv_dataset.encode(test)
        self.assertListEqual(output, expected, msg = 'Encoding failed.')

    def test_split(self):
        """Test splitting functionality."""
        expected = [3, 1]  # Lengths of the respective splits
        train, test = self.csv_dataset.split(self.train, 0.8)
        output = [len(train), len(test)]
        self.assertListEqual(expected, output, msg = 'Splitting with just int failed.')

        expected = [3, 1]
        train, test = self.csv_dataset.split(self.train, [0.8, 0.2])
        output = [len(train), len(test)]
        self.assertListEqual(expected, output, msg = 'Two split values in list failed.')

        expected = [2, 1, 1]
        train, dev, test = self.csv_dataset.split(self.train, [0.6, 0.2, 0.1])
        output = [len(train), len(dev), len(test)]
        self.assertListEqual(expected, output, msg = 'Three split values in list failed.')


class TestDataPoint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up class with data as well."""
        fields = [Field('text', train = True, label = False, ignore = False, ix = 0, cname = 'text'),
                  Field('label', train = False, label = True, cname = 'label', ignore = False, ix = 1)]

        cls.dataset = GeneralDataset(data_dir = '~/PhD/projects/active/Generalisable_abuse/gen/shared/tests/',
                                         ftype = 'csv', fields = fields, train = 'train.csv', dev = None,
                                         test = 'test.csv', train_labels = None, tokenizer = lambda x: x.split(),
                                         lower = True, preprocessor = None, transformations = None,
                                         label_processor = None, sep = ',')
        cls.dataset.load('train')
        cls.train = cls.dataset.data

    def test_datapoint_creation(self):
        """Test that datapoints are created consistently."""
        expected = [{'text': 'me gusta comer en la cafeteria'.lower().split(), 'label': 'SPANISH'},
                    {'text': 'Give it to me'.lower().split(), 'label': 'ENGLISH'},
                    {'text': 'No creo que sea una buena idea'.lower().split(), 'label': 'SPANISH'},
                    {'text': 'No it is not a good idea to get lost at sea'.lower().split(), 'label': 'ENGLISH'}
                    ]
        for exp, out in zip(expected, self.train):
            self.assertDictEqual(exp, out.__dict__, msg = "A dictionary is not created right.")
            self.assertIsInstance(out, Datapoint)

    def test_datapoint_counts(self):
        """Test the correct number of datapoints are created."""
        self.assertEqual(4, len(self.train))
