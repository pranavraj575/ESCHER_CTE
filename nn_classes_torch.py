import torch
from torch import nn


class SkipDense(nn.Module):
    """Dense Layer with skip connection."""

    def __init__(self, num_features, **kwargs):
        super().__init__(**kwargs)
        self.hidden = nn.Linear(in_features=num_features, out_features=num_features)

    def forward(self, x):
        return self.hidden(x) + x


class DefaultNetwork(nn.Module):
    """Implements network as an MLP.

    Implements the network as an MLP with skip connections in
    adjacent layers with the same number of units, except for the last hidden
    connection where a layer normalization is applied.
    """

    def __init__(self, input_size, network_layers, output_dim, activation="leakyrelu", **kwargs):
        super().__init__(**kwargs)
        # self._input_size = input_size
        # self._output_dim = output_dim
        # self._network_layers = network_layers
        # self._activation_name = activation
        # self._kwargs = kwargs

        Activation, act_kwargs = None, None
        if activation == "leakyrelu":
            Activation = nn.LeakyReLU
            act_kwargs = dict(negative_slope=0.2)
        elif activation == "relu":
            Activation = nn.ReLU
            act_kwargs = dict()
        else:
            raise Exception("unknown activation", activation)

        layers = []
        prevunits = input_size
        for units in network_layers[:-1]:
            if prevunits == units:
                layers.append(SkipDense(units))
            else:
                layers.append(nn.Linear(in_features=prevunits, out_features=units))
            layers.append(Activation(**act_kwargs))
            prevunits = units

        layers.append(nn.LayerNorm(normalized_shape=prevunits))
        layers.append(nn.Linear(in_features=prevunits, out_features=network_layers[-1]))
        layers.append(Activation(**act_kwargs))
        layers.append(nn.Linear(in_features=network_layers[-1], out_features=output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        """Applies Regret Network.

        Args:
            inputs: Tuple representing (info_state, legal_action_mask)

        Returns:
            Cumulative regret for each info_state action
        """
        x, mask = inputs
        x = self.network(x)
        return x


class PolicyNetwork(DefaultNetwork):
    """Implements the policy network as an MLP.

    Implements the policy network as a MLP with skip connections in adjacent
    layers with the same number of units, except for the last hidden connection
    where a layer normalization is applied.
    """

    def __init__(
        self,
        input_size,
        policy_network_layers,
        num_actions,
        activation="leakyrelu",
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            network_layers=policy_network_layers,
            output_dim=num_actions,
            activation=activation,
            **kwargs,
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        """Applies Policy Network.

        Args:
            inputs: Tuple representing (info_state, legal_action_mask)

        Returns:
            Action probabilities
        """
        _, mask = inputs
        x = super().forward(inputs=inputs)
        x = torch.where(mask == 1, x, -10e20)
        x = self.softmax(x)
        return x


class RegretNetwork(DefaultNetwork):
    """Implements the regret network as an MLP.

    Implements the regret network as an MLP with skip connections in
    adjacent layers with the same number of units, except for the last hidden
    connection where a layer normalization is applied.
    """

    def __init__(
        self,
        input_size,
        regret_network_layers,
        num_actions,
        activation="leakyrelu",
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            network_layers=regret_network_layers,
            output_dim=num_actions,
            activation=activation,
            **kwargs,
        )

    def forward(self, inputs):
        """Applies Regret Network.

        Args:
            inputs: Tuple representing (info_state, legal_action_mask)

        Returns:
            Cumulative regret for each info_state action
        """
        _, mask = inputs
        x = super().forward(inputs=inputs)
        x = mask * x

        return x


class ValueNetwork(DefaultNetwork):
    """Implements the history value network as an MLP.

    Implements the history value network as an MLP with skip connections in
    adjacent layers with the same number of units, except for the last hidden
    connection where a layer normalization is applied.
    """

    def __init__(self, input_size, val_network_layers, activation="leakyrelu", **kwargs):
        super().__init__(
            input_size=input_size,
            network_layers=val_network_layers,
            output_dim=1,
            activation=activation,
            **kwargs,
        )


if __name__ == "__main__":
    model = RegretNetwork(input_size=3, regret_network_layers=(32,), activation="relu", num_actions=3)

    learning_rate = 1e-3
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(1000):
        optim.zero_grad()
        criterion = nn.MSELoss()
        stuff = torch.randn(256, 3)
        target = stuff * 3
        preds = model((stuff, 1))
        loss = criterion(preds, target)
        print(loss)
        loss.backward()
        optim.step()
    print(model)
