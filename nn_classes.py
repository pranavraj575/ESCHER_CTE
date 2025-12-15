import tensorflow as tf


class SkipDense(tf.keras.layers.Layer):
    """Dense Layer with skip connection."""

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.hidden = tf.keras.layers.Dense(units, kernel_initializer='he_normal')
        self.output_shape = (units,)

    def call(self, x):
        return self.hidden(x) + x

    def build(self, input_shape):
        self.hidden.build(input_shape)


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

    def build(self, input_shape):
        config = self.get_config()
        shapes = (input_shape,) + config['policy_network_layers']
        for layer, dims in zip(self.hidden + [self.lastlayer, self.out_layer], shapes):
            layer.build((dims,))
        self.normalization.build((shapes[-2],))
        self.softmax.build((config['num_actions'],))


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
        self._regret_network_layers = regret_network_layers
        self._activation_name = activation
        self._kwargs = kwargs

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

    def get_config(self):
        config = super().get_config()

        config.update({
            'input_size': self._input_size,
            'regret_network_layers': self._regret_network_layers,
            'num_actions': self._num_actions,
            'activation': self._activation_name,
        })
        config.update(self._kwargs)
        return config

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

    def build(self, input_shape):
        regret_network_layers = self.get_config()['regret_network_layers']
        shapes = (input_shape,) + regret_network_layers
        for layer, dims in zip(self.hidden + [self.lastlayer, self.out_layer], shapes):
            layer.build((dims,))
        self.normalization.build((shapes[-2],))


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
        self._val_network_layers = val_network_layers
        self._activation_name = activation
        self._kwargs = kwargs

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

    def get_config(self):
        config = super().get_config()

        config.update({
            'input_size': self._input_size,
            'val_network_layers': self._val_network_layers,
            'activation': self._activation_name,
        })
        config.update(self._kwargs)
        return config

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

    def build(self, input_shape):
        val_network_layers = self.get_config()['val_network_layers']
        shapes = (input_shape,) + val_network_layers
        for layer, dims in zip(self.hidden + [self.lastlayer, self.out_layer], shapes):
            layer.build((dims,))
        self.normalization.build((shapes[-2],))


if __name__ == '__main__':
    # Set the device to CPU
    tf.config.set_visible_devices([], 'GPU')
    model = RegretNetwork(input_size=3, regret_network_layers=(10,), num_actions=3)
    model.build(3)
    model.summary()

    learning_rate = 1e-3
    optimizer_value = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for _ in range(1000):
        loss_value = tf.keras.losses.MeanSquaredError()

        full_hist_states = tf.constant(tf.random.normal((256, 3)), name='input')
        outputs = full_hist_states*1

        with tf.GradientTape() as tape:
            preds = model((full_hist_states, tf.ones((256, 3))), training=True)
            main_loss = loss_value(
                outputs, preds, sample_weight=1)
            loss = tf.add_n([main_loss], model.losses)
            print(main_loss.numpy())
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer_value.apply_gradients(
            zip(gradients, model.trainable_variables))
