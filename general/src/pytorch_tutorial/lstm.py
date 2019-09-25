import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb


torch.manual_seed(1)

lstm = nn.LSTM(3, 3)  # Initialise LSTM with input dimension size 3 and output dimension 3.
inputs = [torch.randn(1, 3) for _ in range(5)]

hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))

for i in inputs:
    # Step through the sequence one element at a time, hidden contians the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)

# Alternatively, run the entire sequence at once. First value is all hidden states, second is the last hidden state.
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)


def prep_seq(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype = torch.long)


def seq_to_ix(sequence):
    ix_dict = {}
    for s in sequence:
        if s not in ix_dict:
            ix_dict[s] = len(ix_dict)
    return ix_dict


train_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

word_to_ix = {}
tag_to_ix = {}

for sent, tags in train_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    for tag in tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)

print(word_to_ix)
print(tag_to_ix)

EMBEDDING_DIM = 6
HIDDEN_DIM = 6


class LSTMTagger(nn.Module):

    def __init__(self, hidden_dim, embedding_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dims = hidden_dim

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes embeddings as input and outputs hidden states
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # A linear layer to map from hidden state to tagset
        self.linear = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.embeddings(sentence)  # Get embeddings for the sentence.
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))  # Get all hidden states of the LSTM
        tag_space = self.linear(lstm_out.view(len(sentence), -1))  # Predict the tags
        tag_scores = F.log_softmax(tag_space, dim = 1)  # Get the scores for each tag
        return tag_scores


# Training the model
model = LSTMTagger(HIDDEN_DIM, EMBEDDING_DIM, len(word_to_ix), len(tag_to_ix))
loss_func = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)

# Check out scores before training
# Note element i, j is the score for tag j for word i
with torch.no_grad():
    inputs = prep_seq(train_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)

for epoch in range(300):
    for sentence, tags in train_data:
        # Step 1: Clear gradients
        model.zero_grad()

        # Step 2: Get input ready for the network, i.e. turn them into tensors of word indices
        sent_in = prep_seq(sentence, word_to_ix)
        targets = prep_seq(tags, tag_to_ix)

        # Step 3: Run forward pass
        tag_scores = model(sent_in)

        # Step 4: Compute loss, gradients, and update parameters
        loss = loss_func(tag_scores, targets)
        loss.backward()
        optimizer.step()

# Check out the scores before
with torch.no_grad():
    inputs = prep_seq(train_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)


def prep_seq(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype = torch.long)


char_to_ix = {}

for sent, tags in train_data:
    concat_sent = "".join(sent)
    for char in concat_sent:
        if char not in char_to_ix:
            char_to_ix[char.lower()] = len(char_to_ix)


class LSTMWordChar(nn.Module):

    def __init__(self, hidden_dim, embedding_dim, vocab_size, tagset_size, char_hidden_dim):

        # Create an LSTM for each type and a linear layer to output stuff
        super(LSTMWordChar, self).__init__()
        self.w_hidden_dims = hidden_dim
        self.w_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.w_lstm = nn.LSTM(embedding_dim + char_hidden_dim, hidden_dim)  # char_hidden because we concat word+char
        self.w_hidden = self._init_hidden_(hidden_dim)

        self.c_hidden_dims = char_hidden_dim
        self.c_embeddings = nn.Embedding(26, embedding_dim)
        self.c_lstm = nn.LSTM(embedding_dim, char_hidden_dim)  # 26 characters in the english alphabet
        self.c_hidden = self._init_hidden_(char_hidden_dim)

        self.linear = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence, words):

        char_list = []

        # Prepare the sequences
        for word in words:
            c_embeds = self.c_embeddings(word)
            _, c_last = self.c_lstm(c_embeds.view(len(word), 1, -1))
            char_list.append(c_last[0])
        char_tensor = torch.stack(char_list).view(len(words), -1)  # Turn list of tensors into single tensor

        w_embeds = self.w_embeddings(sentence)
        in_embeds = torch.cat((w_embeds, char_tensor), dim = 1)
        w_out, w_last = self.w_lstm(in_embeds.view(len(sentence), 1, -1), self.w_hidden)

        tag_space = self.linear(w_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim = 1)

        return tag_scores

    def _init_hidden_(self, hidden_dim):
        return (torch.zeros(1, 1, hidden_dim), torch.zeros(1, 1, hidden_dim))

    def _prep_seq(self, seq, to_ix, char = False):
        if not char:
            idxs = [to_ix[w] for w in seq]
        else:
            idxs = [to_ix[w.lower()] for w in seq]
        return torch.tensor(idxs, dtype = torch.long)


HIDDEN_DIM = 8
EMBEDDING_DIM = 8
C_EMBEDDING_DIM = 8
C_HIDDEN_DIM = 5

model = LSTMWordChar(HIDDEN_DIM, EMBEDDING_DIM, len(word_to_ix), len(tag_to_ix), C_HIDDEN_DIM)
loss_func = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)

# Train the model
for epoch in range(300):
    for sentence, tags in train_data:
        # Step 1: Clear gradients
        model.zero_grad()

        # Step 2: Get input ready for the network, i.e. turn them into tensors of word indices
        char_seq = [model._prep_seq(w, char_to_ix, char = True) for w in sentence]
        word_seq = model._prep_seq(sentence, word_to_ix, char = False)
        targets = prep_seq(tags, tag_to_ix)

        # Step 3: Run forward pass
        tag_scores = model(word_seq, char_seq)

        # Step 4: Compute loss, gradients, and update parameters
        loss = loss_func(tag_scores, targets)
        loss.backward()
        optimizer.step()


# Test the model
ix_to_tag = {tag_to_ix[tag]: tag for tag in tag_to_ix.keys()}
with torch.no_grad():
    for i in range(2):

        # Get data
        sentence = train_data[i][0]
        print(sentence)

        # Prepare inputs
        sent_in = model._prep_seq(sentence, word_to_ix, char = False)
        word_in = [model._prep_seq(word, char_to_ix, char = True) for word in sentence]

        tag_scores = model(sent_in, word_in)

        tags = torch.argmax(tag_scores, dim = 1).numpy()
        print([ix_to_tag[ix] for ix in tags])
