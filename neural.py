import torch
import torch.nn as nn
import torch.nn.Functional as F
import src.shared.types as t


class LSTMClassifier(nn.Module):

    def __init(self, hidden_dim: int, input_dim: int, no_classes: int, no_layers: int):
        """Initialise the LSTM.
        :param hidden_dim: The dimensionality of the hidden dimension.
        :param input_dim: The dimensionality of the input to the embedding generation.
        :param no_classes: Number of classes for to predict on.
        :param no_layers: The number of recurrent layers in the LSTM (1-3).
        """
        super(LSTMClassifier, self).__init__()

        # Initialise the hidden dim
        self.hidden_dim = hidden_dim

        # Define layers of the network
        self.from_input = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers = no_layers)
        self.to_output = nn.Linear(hidden_dim, no_classes)

        # Set the method for producing "probability" distribution.
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, sequence):
        """The forward step in the classifier.
        :param sequence: The sequence to pass through the network.
        :return scores: The "probability" distribution for the classes.
        """

        out = self.from_input(sequence)  # Get embedding for the sequence
        out, last_layer = self.lstm(out)  # Get layers of the LSTM
        class_scores = self.to_output(out.view(len(sequence), -1))
        prob_dist = self.softmax(class_scores)  # The probability distribution

        return prob_dist


class CNNClassifier(nn.Module):

    def __init__(self, window_sizes: t.List[int], no_filters: int, max_feats: int, embedding_dim: int, no_classes: int):
        """Initialise the model.
        :param window_sizes: The size of the filters (e.g. 1: unigram, 2: bigram, etc.)
        :param no_filters: The number of filters to apply.
        :param max_feats: The maximum length of the sequence to consider.
        """
        self.embedding = nn.Embedding(max_feats, embedding_dim)
        self.conv = [nn.ModuleList(nn.Conv2d(1, no_filters, (w, embedding_dim)) for w in window_sizes)]
        self.linear = nn.Linear(len(window_sizes) * no_filters, no_classes)

    def forward(self, sequence):
        """The forward step of the model.
        :param sequence: The sequence to be predicted on.
        :return scores: The scores computed by the model.
        """
        emb = self.embedding(sequence)  # Get embeddings for sequence
        emb = emb.unsqueeze(1)
        output = [F.relu(conv(sequence)).squeeze(3) for conv in self.conv]
        output = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in sequence]
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

    def forward(self, inputs, hidden):
        """The forward step in the network.
        :param inputs: The inputs to pass through network.
        :param hidden: The hidden representation at the previous timestep.
        :return softmax, hidden: Return the "probability" distribution and the new hidden representation.
        """
        concat_input = torch.cat((inputs, hidden), dim = 1)  # Concatenate input with prev hidden layer
        hidden = self.input2hidden(concat_input)  # Map from input to hidden representation
        output = self.hidden2output(hidden)  # Map from hidden representation to output
        softmax = self.softmax(output)  # Generate probability distribution of output

        return softmax, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_dim)
