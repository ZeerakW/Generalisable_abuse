import torch
import torch.nn as nn


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
