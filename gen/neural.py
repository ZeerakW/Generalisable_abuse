import torch
import torch.nn as nn
import gen.shared.custom_types as t
import torch.nn.functional as F


class LSTMClassifier(nn.Module):

    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int, no_classes: int, no_layers: int):
        """Initialise the LSTM.
        :param input_dim: The dimensionality of the input to the embedding generation.
        :param hidden_dim: The dimensionality of the hidden dimension.
        :param no_classes: Number of classes for to predict on.
        :param no_layers: The number of recurrent layers in the LSTM (1-3).
        """
        super(LSTMClassifier, self).__init__()

        self.itoh = nn.Linear(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, no_layers)
        self.htoo = nn.Linear(hidden_dim, no_classes)

        # Set the method for producing "probability" distribution.
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, sequence, train_mode = False):
        """The forward step in the classifier.
        :param sequence: The sequence to pass through the network.
        :return scores: The "probability" distribution for the classes.
        """
        sequence = sequence.float()
        out = self.itoh(sequence)  # Get embedding for the sequence
        out, last_layer = self.lstm(out)  # Get layers of the LSTM
        out = self.htoo(last_layer[0])
        prob_dist = self.softmax(out)  # The probability distribution

        return prob_dist.squeeze(0)


class MLPClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.2):
        """Initialise the model.
        :param input_dim: The dimension of the input to the model.
        :param hidden_dim: The dimension of the hidden layer.
        :param output_dim: The dimension of the output layer (i.e. the number of classes).
        """
        super(MLPClassifier, self).__init__()

        self.itoh = nn.Linear(input_dim, hidden_dim)
        self.htoh = nn.Linear(hidden_dim, hidden_dim)
        self.htoo = nn.Linear(hidden_dim, output_dim)

        # Set dropout and non-linearity
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, sequence, train_mode = False):

        sequence = sequence.float()
        dropout = self.dropout if train_mode else lambda x: x
        out = dropout(self.tanh(self.itoh(sequence)))
        out = dropout(self.tanh(self.htoh(out)))
        out = out.mean(0)
        out = self.htoo(out)
        prob_dist = self.softmax(out)  # Re-shape to fit batch size.

        return prob_dist


class CNNClassifier(nn.Module):

    def __init__(self, window_sizes: t.List[int], num_filters: int, max_feats: int, hidden_dim: int, no_classes: int,
                 batch_first = False):
        """Initialise the model.
        :param window_sizes: The size of the filters (e.g. 1: unigram, 2: bigram, etc.)
        :param no_filters: The number of filters to apply.
        :param max_feats: The maximum length of the sequence to consider.
        """
        super(CNNClassifier, self).__init__()
        self.batch_first = batch_first

        self.itoh = nn.Linear(max_feats, hidden_dim)  # Works
        self.conv = nn.ModuleList([nn.Conv2d(1, num_filters, (w, hidden_dim)) for w in window_sizes])
        self.linear = nn.Linear(len(window_sizes) * num_filters, no_classes)
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, sequence, train_mode = False):
        """The forward step of the model.
        :param sequence: The sequence to be predicted on.
        :return scores: The scores computed by the model.
        """

        # CNNs expect batch first so let's try that
        if not self.batch_first:
            sequence = sequence.view(sequence.shape[1], sequence.shape[0], sequence.shape[2])
        sequence = sequence.float()
        emb = self.itoh(sequence)  # Get embeddings for sequence
        output = [F.relu(conv(emb.unsqueeze(1))).squeeze(3) for conv in self.conv]
        output = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in output]
        output = torch.cat(output, 1)
        scores = self.softmax(self.linear(output))

        return scores


class RNNClassifier(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """Initialise the RNN classifier.
        :param input_dim: The dimension of the input to the network.
        :param hidden_dim: The dimension of the hidden representation.
        :param output_dim: The dimension of the output representation.
        """
        super(RNNClassifier, self).__init__()

        # Initialise the hidden dim
        self.hidden_dim = hidden_dim

        # Define layers of the network
        self.itoh = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.RNN(hidden_dim, hidden_dim)
        self.htoo = nn.Linear(hidden_dim, output_dim)

        # Set the method for producing "probability" distribution.
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sequence, train_mode = False):
        """The forward step in the network.
        :param inputs: The inputs to pass through network.
        :param hidden: The hidden representation at the previous timestep.
        :return softmax, hidden: Return the "probability" distribution and the new hidden representation.
        """
        sequence = sequence.float()
        hidden = self.itoh(sequence)  # Map from input to hidden representation
        hidden, last_h = self.rnn(hidden)
        output = self.htoo(last_h)  # Map from hidden representation to output
        softmax = self.softmax(output)  # Generate probability distribution of output

        return softmax.squeeze(0)
