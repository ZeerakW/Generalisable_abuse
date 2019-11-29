import unittest
from string import punctuation
from src.shared.data import Field, GeneralDataset, Datapoint
from src.shared import types


class TestDataSet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up class with data as well."""
        cls.data = [("me gusta comer en la cafeteria", "SPANISH"),
                    ("Give it to me", "ENGLISH"),
                    ("No creo que sea una buena idea", "SPANISH"),
                    ("No it is not a good idea to get lost at sea", "ENGLISH")]
        fields = [Field('text', train = True, label = False, ignore = False, ix = 0, cname = 'text'),
                  Field('label', train = True, label = True)]

        cls.csv_dataset = GeneralDataset(data_dir = '~/PhD/projects/active/Generalisable_abuse/gen/shared/tests/',
                                         ftype = 'csv', fields = fields, train = 'train.csv', dev = None,
                                         test = 'test.csv', train_labels = None, tokenizer = lambda x: x.split(),
                                         lower = True, preprocessor = None, transformations = None,
                                         label_processor = None)
        cls.json_dataset = GeneralDataset(data_dir = '~/PhD/projects/active/Generalisable_abuse/gen/shared/tests/',
                                          ftype = 'json', fields = fields, train = 'train.json', dev = None,
                                          test = 'test.json', train_labels = None, tokenizer = lambda x: x.split(),
                                          lower = True, preprocessor = None, transformations = None,
                                          label_processor = None)

    def test_load(self):
        """Test dataset loading."""
        expected = [("me gusta comer en la cafeteria", "SPANISH"),
                    ("Give it to me", "ENGLISH"),
                    ("No creo que sea una buena idea", "SPANISH"),
                    ("No it is not a good idea to get lost at sea", "ENGLISH")]
        csv_train = self.csv_dataset.load('train', skip_header = True)
        csv_output = [(doc.text, doc.label) for doc in csv_train]
        self.assertListEqual(csv_output, expected)

        json_train = self.json_dataset.load('train', skip_header = True)
        json_output = [(doc.text, doc.label) for doc in json_train]
        self.assertListEqual(json_output, expected)

    def test_build_vocab(self):
        """Test vocab building method."""
        expected = """me gusta comer en la cafeteria Give it to me
                   No creo que sea una buena idea No it is not a good idea to get lost at sea""".split()
        train = self.csv_dataset.load('train')
        self.csv_dataset.build_label_vocab(train)
        self.assertListEqual(list(sorted(self.csv_dataset.stoi.keys())), list(sorted(expected)),
                             msg = "Vocab building failed.")

    def test_extend_vocab(self):
        """Test extending vocab."""
        train = """me gusta comer en la cafeteria Give it to me
                No creo que sea una buena idea No it is not a good idea to get lost at sea""".split()
        test = "Yo creo que si it is lost on me".split()
        expected = train + test
        train = self.csv_dataset.load('train')
        test = self.csv_dataset.load('test')
        self.csv_dataset.build_vocab(train)
        self.csv_dataset.extend_vocab(test)
        self.assertListEqual(list(self.csv_dataset.stoi.keys()), expected, msg = "Vocab Extension Failed.")

    def test_vocab_token_lookup(self):
        """Test looking up in vocab."""
        train = self.csv_dataset.load('train')
        self.csv_dataset.build_label_vocab(train)
        self.assertEqual(self.csv_dataset.vocab_token_lookup('me'), 0, msg = "Vocab token lookup failed.")

    def test_vocab_ix_lookup(self):
        """Test looking up in vocab."""
        train = self.csv_dataset.load('train')
        self.csv_dataset.build_label_vocab(train)
        self.assertEqual(self.csv_dataset.vocab_ix_lookup(0), 'me', msg = "Vocab ix lookup failed.")

    def test_vocab_size(self):
        """Test vocab size is expected size."""
        train = self.csv_dataset.load('train')
        self.csv_dataset.build_vocab(train)
        self.assertEqual(self.csv_dataset.vocab_size(), 23)

    def test_build_label_vocab(self):
        """Test building label vocab."""
        train = self.csv_dataset.load('train')
        self.csv_dataset.build_label_vocab(train)
        self.assertListEqual(list(self.csv_dataset.ltoi.keys()), ['SPANISH', 'ENGLISH'])

    def test_label_name_lookup(self):
        """Test looking up in label."""
        train = self.csv_dataset.load('train')
        self.csv_dataset.build_label_label(train)
        self.assertEqual(self.csv_dataset.label_name_lookup('SPANISH'), 0, msg = "label name lookup failed.")

    def test_label_ix_lookup(self):
        """Test looking up in label."""
        train = self.csv_dataset.load('train')
        self.csv_dataset.build_label_label(train)
        self.assertEqual(self.csv_dataset.label_ix_lookup(0), 'SPANISH', msg = "label ix lookup failed.")

    def test_label_count(self):
        train = self.csv_dataset.load('train')
        self.csv_dataset.build_label(train)
        self.assertEqual(self.csv_dataset.label_count(), 2, msg = 'Test that label count matches labels failed.')

    def test_process_label(self):
        """Test label processing."""
        train = self.csv_dataset.load('train')
        self.csv_dataset.build_label_vocab(train)
        expected = 0
        result = self.csv_dataset.process_label('SPANISH')
        self.assertEqual(result, expected, msg = "Labelprocessor failed without custom processor")

        expected = 1
        result = self.csv_dataset.process_label('SPANISH', processor = lambda x: 1)
        self.assertEqual(result, expected, msg = "Labelprocessor failed with custom processor")

    def test_load_label(self):
        self.assertEqual(True, False)
