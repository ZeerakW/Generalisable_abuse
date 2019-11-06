import pdb
import torch
import torch.nn as nn
import gen.shared.types as t
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

    def forward(self, sequence):
        """The forward step in the classifier.
        :param sequence: The sequence to pass through the network.
        :return scores: The "probability" distribution for the classes.
        """
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

        self.input2hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden2hidden = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2output = nn.Linear(hidden_dim, output_dim)

        # Set dropout and non-linearity
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, sequence, train_mode = False):

        sequence = sequence.float()
        dropout = self.dropout if train_mode else lambda x: x
        out = dropout(self.tanh(self.input2hidden(sequence)))
        out = dropout(self.tanh(self.hidden2hidden(out)))
        out = self.hidden2output(out)
        prob_dist = self.softmax(out.view(1, -1))  # Re-shape to only address the last model.

        return prob_dist.squeeze(1)


class CNNClassifier(nn.Module):

    def __init__(self, window_sizes: t.List[int], no_filters: int, max_feats: int, hidden_dim: int, no_classes: int):
        """Initialise the model.
        :param window_sizes: The size of the filters (e.g. 1: unigram, 2: bigram, etc.)
        :param no_filters: The number of filters to apply.
        :param max_feats: The maximum length of the sequence to consider.
        """
        super(CNNClassifier, self).__init__()

        self.embedding = nn.Linear(max_feats, hidden_dim)
        self.conv = nn.ModuleList([nn.Conv2d(1, no_filters, (w, hidden_dim)) for w in window_sizes])
        self.linear = nn.Linear(len(window_sizes) * no_filters, no_classes)

    def forward(self, sequence):
        """The forward step of the model.
        :param sequence: The sequence to be predicted on.
        :return scores: The scores computed by the model.
        """
        pdb.set_trace()
        emb = self.embedding(sequence)  # Get embeddings for sequence
        # emb = emb.unsqueeze(1)
        output = [F.relu(conv(emb)).squeeze(3) for conv in self.conv]
        output = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in output]
        output = torch.cat(output, 1)
        scores = self.linear(output)

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
        self.input2hidden = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.hidden2output = nn.Linear(hidden_dim, output_dim)

        # Set the method for producing "probability" distribution.
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden = []):
        """The forward step in the network.
        :param inputs: The inputs to pass through network.
        :param hidden: The hidden representation at the previous timestep.
        :return softmax, hidden: Return the "probability" distribution and the new hidden representation.
        """
        concat_input = torch.cat((inputs.view(1, -1).float(), hidden), dim = 1)  # Concatenate input with prev hidden layer
        hidden = self.input2hidden(concat_input)  # Map from input to hidden representation
        output = self.hidden2output(hidden)  # Map from hidden representation to output
        softmax = self.softmax(output)  # Generate probability distribution of output

        return softmax, hidden

    def init_hidden(self, shape):
        # return torch.zeros(shape[0], shape[1], self.hidden_dim)
        return torch.zeros(shape[0], self.hidden_dim)
