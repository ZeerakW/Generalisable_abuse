import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

word_to_ix = {'hello': 0, 'world': 1}
embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.tensor([word_to_ix['hello']], dtype = torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)

# N-gram languge modeling
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))  # What does view do?
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr = 0.050)

for epoch in range(10):
    total_loss = 0
    for context, target in trigrams:
        # Step 1: Prepare input to be passed to the modal
        # (turn the words into integer indices and wrap in tensor)
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype = torch.long)

        # Step 2: Zero out gradients from the previous instance
        model.zero_grad()

        # Step 3. Run forward pass to get log probs for next words
        log_probs = model(context_idxs)

        # Step 4. Compute loss
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype = torch.long))

        # Step 5. Backprop
        loss.backward()
        optimizer.step()

        # Update total loss
        total_loss += loss.item()
    losses.append(total_loss)
print(losses)


# Continuous Bag of Words
CONTEXT_SIZE = 2  # 2 words to either side of the current word
embedding_dim = 10
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

vocab = set(raw_text)
vocab_size = len(vocab)
word_to_ix, ix_to_word = {}, {}

for i, word in enumerate(vocab):
    word_to_ix[word] = i
    ix_to_word[i] = word

data = []

for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1], raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))


class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim = 1)
        return log_probs

    def get_word_embedding(self, word):
        word = torch.LongTensor([word_to_ix[word]])
        return self.embeddings(word).view((1, -1))


losses = []
model = CBOW(len(vocab), EMBEDDING_DIM)
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


for epoch in range(100):
    total_loss = 0
    for context, target in data:
        context_vector = make_context_vector(context, word_to_ix)  # Step 1
        model.zero_grad()  # Step 2
        log_probs = model(context_vector)  # Step 3
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype = torch.long))  # Step 4

        # Step 5
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    losses.append(total_loss)

print(losses)


def get_index_of_max(input):
    index = 0
    for i in range(1, len(input)):
        if input[i] > input[index]:
            index = i
    return index


def get_max_prob_result(input, ix_to_word):
    return ix_to_word[get_index_of_max(input)]


# Test
context = ['People', 'create', 'to', 'direct']
context_vector = make_context_vector(context, word_to_ix)
a = model(context_vector).data.numpy()
print('Raw text: {}\n'.format(' '.join(raw_text)))
print('Context: {}\n'.format(context))
print('Prediction: {}'.format(get_max_prob_result(a[0], ix_to_word)))
