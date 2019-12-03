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

    # @classmethod
    # def tearDown(cls):
    #     """Tear down class."""
    #     cls.csv_dataset = 0
    #     cls.json_dataset = 0

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

    def test_build_vocab(self):
        """Test vocab building method."""
        expected = list(sorted("""me gusta comer en la cafeteria Give it to me
                   No creo que sea una buena idea No it is not a good idea to get lost at sea""".split()))
        self.csv_dataset.build_token_vocab(self.train)
        output = list(sorted(self.csv_dataset.stoi.keys()))
        self.assertListEqual(output, expected, msg = 'Vocab building failed.')

    def test_extend_vocab(self):
        """Test extending vocab."""
        train = """me gusta comer en la cafeteria Give it to me
                No creo que sea una buena idea No it is not a good idea to get lost at sea""".split()
        test = "Yo creo que si it is lost on me".split()
        expected = train + test
        test = self.csv_dataset.load('test')
        self.csv_dataset.build_vocab(self.train)
        self.csv_dataset.extend_vocab(test)
        output = list(self.csv_dataset.stoi.keys())
        self.assertListEqual(output, expected, msg = 'Vocab Extension Failed.')

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
        self.csv_dataset.build_vocab(self.train)
        output = self.csv_dataset.vocab_size()
        expected = 23
        self.assertEqual(output, expected, msg = 'Building vocab failed.')

    def test_build_label_vocab(self):
        """Test building label vocab."""
        self.csv_dataset.build_label_vocab(self.train)
        output = list(sorted(self.csv_dataset.ltoi.keys()))
        expected = ['ENGLISH', 'SPANISH']
        self.assertListEqual(output, expected, msg = 'Building label vocab failed.')

    def test_label_name_lookup(self):
        """Test looking up in label."""
        self.csv_dataset.build_label_label(self.train)
        output = self.csv_dataset.label_name_lookup('SPANISH')
        expected = 0
        self.assertEqual(output, expected, msg = 'label name lookup failed.')

    def test_label_ix_lookup(self):
        '''Test looking up in label.'''
        self.csv_dataset.build_label_label(self.train)
        output = self.csv_dataset.label_ix_lookup(0)
        expected = 'SPANISH'
        self.assertEqual(output, expected, msg = 'label ix lookup failed.')

    def test_label_count(self):
        """Test label size is expected."""
        self.csv_dataset.build_label(self.train)
        expected = self.csv_dataset.label_count()
        output = 2
        self.assertEqual(output, expected, msg = 'Test that label count matches labels failed.')

    def test_process_label(self):
        """Test label processing."""
        self.csv_dataset.build_label_vocab(self.train)
        expected = 0
        output = self.csv_dataset.process_label('SPANISH')
        self.assertEqual(output, expected, msg = 'Labelprocessor failed without custom processor')

        expected = 1
        output = self.csv_dataset.process_label('SPANISH', processor = lambda x: 1)
        self.assertEqual(output, expected, msg = 'Labelprocessor failed with custom processor')

    def test_process_doc(self):
        """Test document processing."""
        setattr(self.csv_dataset, 'lower', False)
        setattr(self.csv_dataset, 'preprocessor', None)
        setattr(self.csv_dataset, 'repr_transform', None)

        inputs = "Give it to me, baby. Uhuh! Uhuh!"
        expected = inputs.split()
        output = self.csv_dataset.process_doc(inputs)
        self.assertListEqual(output, expected, msg = 'Process Document failed.')
        self.assertIsInstance(output, list, msg = 'Process document returned wrong type.')

        expected = inputs.lower().split()
        setattr(self.csv_dataset, 'lower', True)
        output = self.csv_dataset.process_doc(inputs)
        self.assertListEqual(output, expected, msg = 'Process Document failed with lowercasing.')
        self.assertIsInstance(output, list, msg = 'Process document with lowercasing produces wrong type.')

        output = self.csv_dataset.process_doc(inputs.split())
        self.assertListEqual(output, expected, msg = 'Process Document failed with input type list.')
        self.assertIsInstance(output, list, msg = 'Process document with input type list produces wrong type.')

        setattr(self.csv_dataset, 'preprocessor', lambda x: ["TEST" if '!' in tok else tok for tok in x])
        expected = ["TEST" if '!' in tok else tok for tok in inputs.lower().split()]
        output = self.csv_dataset.process_doc(inputs)
        self.assertListEqual(output, expected, msg = 'Process Document failed with preprocessor')
        self.assertIsInstance(output, list, msg = 'Process Document with preprocessor returned wrong type.')

        def transform(doc):
            transform = {'give': 'VERB', 'it': 'DET', 'to': "PREP", 'me': "PRON", 'uhuh!': 'AGREEMENT',
                         'uhuh': 'AGREEMENT'}
            for w in doc:
                yield transform[w.lower()]

        setattr(self.csv_dataset, 'repr_transform', transform)
        expected = "VERB DET PREP PRON AGREEMENT AGREEMENT".split()
        output = self.csv_dataset.process_doc(inputs)
        self.assertListEqual(output, expected, msg = 'Process Document failed with represntation transformation.')
        self.assertIsInstance(output, list, msg = 'Process Document with representation transformation returned wrong')

    def test_pad(self):
        """Test padding of document."""
        inputs = "me gusta comer en la cafeteria"
        expected_pad = 4 * ['<pad>'] + inputs.split()
        output_pad = list(self.csv_dataset.pad(self.train, length = 10))
        self.assertListEqual(output_pad, expected_pad, msg = 'Padding doc failed.')

        expected_pad = inputs.split()[:5]
        output_pad = list(self.csv_dataset.pad(self.train, length = 5))
        self.assertListEqual(output_pad, expected_pad, msg = 'Trimming doc failed.')

        expected_pad = inputs.split()
        output_pad = list(self.csv_dataset.pad(self.train, length = len(expected_pad)))
        self.assertListEqual(output_pad, expected_pad, msg = 'Zero padding failed.')

    def test_onehot_encoding(self):
        """Test the onehot encoding."""
        self.csv_dataset.build_label_vocab(self.train)

        test = ["Give it to me".split()]
        expected = [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1]
        output = self.csv_dataset.onehot_encode(test)
        self.assertListEqual(output, expected, msg = 'Onehot encoding failed.')

    def test_encoding(self):
        """Test the encoding."""
        self.csv_dataset.build_label_vocab(self.train)
        test = ["Give it to me".split()]
        expected = [2, 4, 5, 1]
        output = self.csv_dataset.encode(test)
        self.assertListEqual(output, expected, msg = 'Encoding failed.')

    def test_split(self):
        """Test splitting functionality."""
        expected = [4, 1]  # Lengths of the respective splits
        train, test = self.csv_dataset.split(self.train, 0.8)
        output = [len(train), len(test)]
        self.assertListEqual(expected, output, msg = 'Splitting with just int failed.')

        expected = [4, 1]
        train, test = self.csv_dataset.split(self.train, [0.8, 0.2])
        output = [len(train), len(test)]
        self.assertListEqual(expected, output, msg = 'Two split values in list failed.')

        expected = [3, 1, 1]
        train, dev, test = self.csv_dataset.split(self.train, [0.8, 0.1, 0.1])
        output = [len(train), len(dev), len(test)]
        self.assertListEqual(expected, output, msg = 'Three split values in list failed.')
