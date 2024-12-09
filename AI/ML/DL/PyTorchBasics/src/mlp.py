import torch

from src.utils import save, load


class MLP(torch.nn.Module):
    def __init__(
        self,
        number_of_hidden_layers: int,
        input_size: int,
        hidden_size: int,
        activation: torch.nn.Module,
    ):
        """
        Construct a simple MLP
        """
        # NOTE: don't edit this constructor

        super().__init__()
        assert (
            number_of_hidden_layers >= 0
        ), "number_of_hidden_layers must be at least 0"

        dims_in = [input_size] + [hidden_size] * number_of_hidden_layers
        dims_out = [hidden_size] * number_of_hidden_layers + [1]

        layers = []
        for i in range(number_of_hidden_layers + 1):
            layers.append(torch.nn.Linear(dims_in[i], dims_out[i]))

            # No final activation
            if i < number_of_hidden_layers:
                layers.append(activation)

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def initialize(self):
        """
        Initialize all the model's weights.
        See https://pytorch.org/docs/stable/nn.init.html
        """

        # This is one way that I discovered using a apply function to initialize weights
        # def init_weight(layer):
        #     if isinstance(layer, torch.nn.Linear):
        #         torch.nn.init.uniform_(layer.weight)

        # self.net.apply(init_weight)

        # The same process could also be accomplished by iterating
        # over each layer in the network.
        for layer in self.net:
            if isinstance(layer, torch.nn.Linear):
                # gain = torch.nn.init.calculate_gain("relu")
                # torch.nn.init.xavier_uniform_(layer.weight, gain=gain)
                # print(f"layer before initialization: {layer.weight}")
                torch.nn.init.normal_(layer.weight)
                # print(f"layer after initialization: {layer.weight}")
                # I did end up initializing the bias, I think it kind of helped
                layer.bias.data.fill_(0.01)
                # print(f"layer after initializing the bias term: {layer.bias}")

    def save_model(self, filename):
        """
        Use `src.utils.save` to save this model to file.

        Note: You may want to save a dictionary containing the model's state.

        Args
            filename: the file to which to save the model
        """
        save(filename, state_dict=self.state_dict())

    def load_model(self, filename):
        """
        Use `src.utils.load` to load this model from file.

        Note: in addition to simply loading the saved model, you must use the
              information from that checkpoint to update the model's state.

        Args
            filename: the file from which to load the model
        """
        # state dictionary simply maps
        # model's parameters to saved parameters
        checkpoint = load(filename)
        self.load_state_dict(checkpoint["state_dict"])
