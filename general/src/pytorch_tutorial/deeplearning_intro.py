import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(32)

# A = Matrix, b = bias term
# Apply linear transformation to the incoming data (y = A^T x + b)
lin = nn.Linear(5, 3)  # Maps from R^5 to R^3, parameters A, b
data = torch.randn(2, 5)  # Data is 2x5. A maps from 5 to 3.
print("Linear data")
print(lin(data))

# Doing multiple iterations of linear mappings makes no sense, because
# you can just do one with the same result
# To start adding non-linearity, we need to add non-linear functions e.g. TanH, ReLU

print("ReLU results")
data = torch.randn(2,2)
print(F.relu(data))

# Softmax can be used in final non-linear transformation because outputs a probability distribution.
print("Softmax")
data = torch.randn(5)
print(F.softmax(data, dim=0))
print(F.softmax(data, dim=0).sum())
print(F.log_softmax(data, dim=0))

# Logistic Regression Language Classifier

print("Logistic Regression")
data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

test = [("Yo creo que si".split(), "SPANISH"),
        ("it is lost on me".split(), "ENGLISH")]

word_to_ix = {}

for sent, _ in data + test:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2

class BoWClassifier(nn.Module):
    def __init__(self, num_labels, vocab_size):
        super(BoWClassifier, self).__init__()  # Always do this when inherenting from nn.Module

        # Define the parameters needed.
        # Here it's a linear (affine) mapping, so nn.Linear
        # The first arg is the size of the feature vector (all words), so vocab_size
        # The second argument is the number of outputs, namely the number of labels.
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vector):
        # Pass through the linear transformation then do the softmax.
        out = self.linear(bow_vector)
        return F.log_softmax(out, dim=1)


def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))  # Initialise a tensor the size of the vocabulary
    for word in sentence:
        vec[word_to_ix[word]] += 1
    return vec.view(1, -1)


def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])


model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)
for param in model.parameters():
    print(param)

with torch.no_grad(): # Don't need to do any back-propagation since it's linear
    sample = data[0]
    bow_vector = make_bow_vector(sample[0], word_to_ix)
    log_probs = model(bow_vector)
    print(log_probs)

label_to_ix = {"SPANISH": 0, "ENGLISH": 1}

# Now with backprop
print("With Backprop")

# Run on test data for before/after view
with torch.no_grad():
    for instance, label in test:
        bow_vec = make_bow_vector(instance, word_to_ix)
        log_probs = model(bow_vec)
        print(log_probs)

# Print the matrix column corresponding to "creo"
print(next(model.parameters())[:, word_to_ix["creo"]])

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(10):
    for instance, label in data:
        # Step 1: Clear out gradients before each document is parsed
        model.zero_grad()

        # Step 2: Make BoW Vector and wrap the target (label) in a tensor
        bow_vec = make_bow_vector(instance, word_to_ix)
        target = make_target(label, label_to_ix)

        # Step 3: Do the forward pass
        log_probs = model(bow_vec)

        # Step 4: Compute loss, gradients, and update by calling the optimise step
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    for instance, label in test:
        bow_vec = make_bow_vector(instance, word_to_ix)
        log_probs = model(bow_vec)
        print(log_probs)

# Index corresponding to Spanish goes up, English goes down!
print(next(model.parameters())[:, word_to_ix["creo"]])
