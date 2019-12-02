# from torchtext.data import Field, TabularDataset, BucketIterator
import shared.types as types
from shared.prep import Dataset, BatchGenerator

# Initialise the data reader
ds = Dataset(data_dir = '~/Documents/PhD/projects/Generalisable_abuse/data',
             fields = None,
             splits = {'train': 'cyberbullying_dataset.csv', 'test': 'test.csv', 'validation': 'dev.csv'},
             cleaners = ['lower', 'url', 'hashtag', 'username'],
             batch_sizes = (32, 32, 32),
             ftype = 'csv',
             shuffle = True,
             sep = ',',
             repeat_in_batches = True)

# Initialise the field types
text = types.text_data
label = types.text_label
ds.set_field_attribute(text, 'tokenize', ds.tokenize)

# Set fields
fields = [('id', None),
          ('bad_word', text),
          ('question', text),
          ('question_sentiment_gold', label),
          ('answer', text),
          ('answer_sentiment_gold', label),
          ('username', text)]

print(ds.fields)
ds.fields = fields
print(ds.fields)

train, dev, test = ds.load_data()
print(train, test, dev)

text.build_vocab(train)
label.build_vocab(train)

train_iter, dev_iter, test_iter = ds.generate_batches(lambda x: len(x.question))
print(train_iter, dev_iter, test_iter)

# print("Train iter", next(iter(train_iter)))
# print("Test iter", next(iter(test_iter)))
# print("dev iter", next(iter(dev_iter)))
#
train_batches = BatchGenerator(train, 'question', 'question_sentiment_gold')
print("train batch", next(iter(train_batches)))

test_batches = BatchGenerator(test, 'question', 'question_sentiment_gold')
print("test batch", next(iter(test_batches)))

dev_batches = BatchGenerator(dev, 'question', 'question_sentiment_gold')
print("dev batch", next(iter(dev_batches)))


# # Initialise the field types
# text = Field(sequential = True,
#                   include_lengths=True,
#                   use_vocab=True)
# label = Field(sequential = False,
#               include_lengths = False,
#               use_vocab = True,
#               pad_token = None,
#               unk_token = None)
#
# fields = [('id', None),
#           ('bad_word', text),
#           ('question', text),
#           ('question_sentiment_gold', label),
#           ('answer', text),
#           ('answer_sentiment_gold', label),
#           ('username', text)]
#
# train, dev, test = TabularDataset.splits(path = '~/Documents/PhD/projects/Generalisable_abuse/data/',
#                                          format = 'csv',
#                                          fields = fields,
#                                          train = 'cyberbullying_dataset.csv', validation = 'dev.csv', test = 'test.csv',
#                                          skip_header = True)
#
# train_iter, dev_iter, test_iter = BucketIterator.splits((train, dev, test),
#                                                         batch_sizes = (32, 32, 32),
#                                                         sort_key = lambda x: len(x.question),
#                                                         device = 'cpu')
#
# print(train, dev, test)
#
# text.build_vocab(train)
# label.build_vocab(train)
#
# print("Train iter", next(iter(train_iter)))
# print("Test iter", next(iter(test_iter)))
# print("dev iter", next(iter(dev_iter)))
