import torch.nn as nn


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
