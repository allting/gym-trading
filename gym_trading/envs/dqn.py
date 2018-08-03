"""DQN Class

DQN(NIPS-2013)
"Playing Atari with Deep Reinforcement Learning"
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

DQN(Nature-2015)
"Human-level control through deep reinforcement learning"
http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
"""
import numpy as np
import tensorflow as tf


class DQN:

    def __init__(self, session: tf.Session, input_shape: tuple, output_size: int, name: str="main") -> None:
        """ DQN Agent can

        1) Build network
        2) Predict Q_value given state
        3) Train parameters

        Args:
            session (tf.Session): Tensorflow session
            input_shape (int): Input dimension
            output_size (int): Number of discrete actions
            name (str, optional): TF Graph will be built under this name scope
        """
        self.session = session
        self.input_shape = input_shape
        self.output_size = output_size
        self.net_name = name
        self.seq_length = 7
        self.hidden_dim = 150

        self._build_network()

    def _build_network(self, h_size=16, l_rate=0.001) -> None:
        """DQN Network architecture (simple MLP)

        Args:
            h_size (int, optional): Hidden layer dimension
            l_rate (float, optional): Learning rate
        """
        # with tf.variable_scope(self.net_name):
        #     self._X = tf.placeholder(tf.float32, [None, self.input_shape], name="input_x")
        #     net = self._X

        #     net = tf.layers.dense(net, h_size, activation=tf.nn.relu)
        #     net = tf.layers.dense(net, self.output_size)
        #     self._Qpred = net

        #     self._Y = tf.placeholder(tf.float32, shape=[None, self.output_size])
        #     self._loss = tf.losses.mean_squared_error(self._Y, self._Qpred)

        #     optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
        #     self._train = optimizer.minimize(self._loss)

        # with tf.variable_scope(self.net_name):
        #     self._X = tf.placeholder(tf.float32, [None, self.input_shape[0], self.input_shape[1]], name='input_x')

        #     cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_dim, state_is_tuple=True, activation=tf.tanh)
        #     outputs, _states = tf.nn.dynamic_rnn(cell, self._X, dtype=tf.float32)
        #     self._Qpred = tf.contrib.layers.fully_connected(outputs[:, -1], self.output_size, activation_fn=None)

        #     self._Y = tf.placeholder(tf.float32, shape=[None, self.output_size])
        #     self._loss = tf.reduce_sum(tf.square(self._Qpred - self._Y))
            
        #     optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
        #     self._train = optimizer.minimize(self._loss)

        with tf.variable_scope(self.net_name):
            n_layers = 3
            self.state = tf.placeholder(tf.float32, [None, self.input_shape[0], self.input_shape[1]], name='state') # time, batch, in
            self.actions = tf.placeholder(tf.int32, [None], name='actions') #

            # layers = [tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_dim) for _ in range(n_layers)]
            # multi_cell = tf.contrib.rnn.MultiRNNCell(layers)
            # outputs, states = tf.nn.dynamic_rnn(multi_cell, self.state, dtype=tf.float32)

            # hidden_state = states[-1][1]

            cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_dim)
            self.outputs, states = tf.nn.dynamic_rnn(cell, self.state, dtype=tf.float32)

            self.logits = tf.layers.dense(states[1], self.output_size, name='softmax')
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions, logits=self.logits)
            self._loss = tf.reduce_mean(xentropy, name='loss')

            optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
            self._train = optimizer.minimize(self._loss)
            self._correct = tf.nn.in_top_k(self.logits, self.actions, 1)
            self._accuracy = tf.reduce_mean(tf.cast(self._correct, tf.float32))


    def predict(self, state: np.ndarray) -> np.ndarray:
        """Returns Q(s, a)

        Args:
            state (np.ndarray): State array, shape (n, input_dim)

        Returns:
            np.ndarray: Q value array, shape (n, output_dim)
        """
        try:
            x = np.reshape(state, [-1, self.input_shape[0], self.input_shape[1]])
        except Exception as e:
            print('Error:{}'.format(e))
        
        # return self.session.run(self._train, feed_dict={self.state: x})
        return self.logits.eval(feed_dict={self.state: x})
        # return np.argmax(result, axis=1)
        # return tf.argmax(result, 1)

    def update(self, x_stack: np.ndarray, y_stack: np.ndarray) -> list:
        """Performs updates on given X and y and returns a result

        Args:
            x_stack (np.ndarray): State array, shape (n, input_dim)
            y_stack (np.ndarray): Target Q array, shape (n, output_dim)

        Returns:
            list: First element is loss, second element is a result from train step
        """

        x_data = np.reshape(x_stack, [-1, self.input_shape[0], self.input_shape[1]])
        # y_data = np.reshape(y_stack, [-1, self.output_size])
        feed = {
            self.state: x_data,
            self.actions: y_stack
        }
        # return self.session.run(self._train, feed)
        return self.session.run([self._loss, self._train], feed)
