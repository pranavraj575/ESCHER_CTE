import tensorflow as tf


class SkipDense(tf.keras.layers.Layer):
    """Dense Layer with skip connection."""

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.hidden = tf.keras.layers.Dense(units, kernel_initializer='he_normal')

    def call(self, x):
        return self.hidden(x) + x


class PolicyNetwork(tf.keras.Model):
    """Implements the policy network as an MLP.

    Implements the policy network as a MLP with skip connections in adjacent
    layers with the same number of units, except for the last hidden connection
    where a layer normalization is applied.
    """

    def __init__(self,
                 input_size,
                 policy_network_layers,
                 num_actions,
                 activation='leakyrelu',
                 **kwargs):
        super().__init__(**kwargs)
        self._input_size = input_size
        self._num_actions = num_actions
        if activation == 'leakyrelu':
            self.activation = tf.keras.layers.LeakyReLU(alpha=0.2)
        elif activation == 'relu':
            self.activation = tf.keras.layers.ReLU()
        else:
            self.activation = activation
        # (PR) added these so we can use them in self.get_config()
        self._policy_network_layers = policy_network_layers
        self._activation_name = activation
        self._kwargs = kwargs
        self.softmax = tf.keras.layers.Softmax()

        self.hidden = []
        prevunits = 0
        for units in policy_network_layers[:-1]:
            if prevunits == units:
                self.hidden.append(SkipDense(units))
            else:
                self.hidden.append(
                    tf.keras.layers.Dense(units, kernel_initializer='he_normal'))
            prevunits = units
        self.normalization = tf.keras.layers.LayerNormalization()
        self.lastlayer = tf.keras.layers.Dense(
            policy_network_layers[-1], kernel_initializer='he_normal')

        self.out_layer = tf.keras.layers.Dense(num_actions)

    def get_config(self):
        # (PR) added this method so model.save() works
        config = super().get_config()

        config.update({
            'input_size': self._input_size,
            'policy_network_layers': self._policy_network_layers,
            'num_actions': self._num_actions,
            'activation': self._activation_name,
        })
        config.update(self._kwargs)
        return config

    @tf.function
    def call(self, inputs):
        """Applies Policy Network.

        Args:
            inputs: Tuple representing (info_state, legal_action_mask)

        Returns:
            Action probabilities
        """
        x, mask = inputs
        for layer in self.hidden:
            x = layer(x)
            x = self.activation(x)

        x = self.normalization(x)
        x = self.lastlayer(x)
        x = self.activation(x)
        x = self.out_layer(x)
        x = tf.where(mask == 1, x, -10e20)
        x = self.softmax(x)
        return x


class RegretNetwork(tf.keras.Model):
    """Implements the regret network as an MLP.

    Implements the regret network as an MLP with skip connections in
    adjacent layers with the same number of units, except for the last hidden
    connection where a layer normalization is applied.
    """

    def __init__(self,
                 input_size,
                 regret_network_layers,
                 num_actions,
                 activation='leakyrelu',
                 **kwargs):
        super().__init__(**kwargs)
        self._input_size = input_size
        self._num_actions = num_actions
        if activation == 'leakyrelu':
            self.activation = tf.keras.layers.LeakyReLU(alpha=0.2)
        elif activation == 'relu':
            self.activation = tf.keras.layers.ReLU()
        else:
            self.activation = activation

        self.hidden = []
        prevunits = 0
        for units in regret_network_layers[:-1]:
            if prevunits == units:
                self.hidden.append(SkipDense(units))
            else:
                self.hidden.append(
                    tf.keras.layers.Dense(units, kernel_initializer='he_normal'))
            prevunits = units
        self.normalization = tf.keras.layers.LayerNormalization()
        self.lastlayer = tf.keras.layers.Dense(
            regret_network_layers[-1], kernel_initializer='he_normal')

        self.out_layer = tf.keras.layers.Dense(num_actions)

    @tf.function
    def call(self, inputs):
        """Applies Regret Network.

        Args:
            inputs: Tuple representing (info_state, legal_action_mask)

        Returns:
            Cumulative regret for each info_state action
        """
        x, mask = inputs
        for layer in self.hidden:
            x = layer(x)
            x = self.activation(x)

        x = self.normalization(x)
        x = self.lastlayer(x)
        x = self.activation(x)
        x = self.out_layer(x)
        x = mask*x

        return x


class ValueNetwork(tf.keras.Model):
    """Implements the history value network as an MLP.

    Implements the history value network as an MLP with skip connections in
    adjacent layers with the same number of units, except for the last hidden
    connection where a layer normalization is applied.
    """

    def __init__(self,
                 input_size,
                 val_network_layers,
                 activation='leakyrelu',
                 **kwargs):
        super().__init__(**kwargs)
        self._input_size = input_size
        if activation == 'leakyrelu':
            self.activation = tf.keras.layers.LeakyReLU(alpha=0.2)
        elif activation == 'relu':
            self.activation = tf.keras.layers.ReLU()
        else:
            self.activation = activation

        self.hidden = []
        prevunits = 0
        for units in val_network_layers[:-1]:
            if prevunits == units:
                self.hidden.append(SkipDense(units))
            else:
                self.hidden.append(
                    tf.keras.layers.Dense(units, kernel_initializer='he_normal'))
            prevunits = units
        self.normalization = tf.keras.layers.LayerNormalization()
        self.lastlayer = tf.keras.layers.Dense(
            val_network_layers[-1], kernel_initializer='he_normal')

        self.out_layer = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, inputs):
        """Applies Value Network.

        Args:
            inputs: Tuple representing (info_state, legal_action_mask)

        Returns:
            Cumulative regret for each info_state action
        """
        x, mask = inputs
        for layer in self.hidden:
            x = layer(x)
            x = self.activation(x)

        x = self.normalization(x)
        x = self.lastlayer(x)
        x = self.activation(x)
        x = self.out_layer(x)

        return x


if __name__ == '__main__':
    p = PolicyNetwork(input_size=10, policy_network_layers=(128, 64, 64,), num_actions=3)
    p.build(10)
    p.summary()
    from tensorflow.keras.utils import plot_model
