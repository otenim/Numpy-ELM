import numpy as np

def _mean_squared_error(y_true, y_pred):
    return 0.5 * np.mean((y_true - y_pred)**2)

def _mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def _identity(x):
    return x

class ELM(object):

    def __init(
        self, n_input_nodes, n_hidden_nodes, n_output_nodes,
        activation='sigmoid', loss='mean_squared_error', name=None,
        beta_init=None, alpha_init=None, bias_init=None):

        self.name = name
        self.__n_input_nodes = n_input_nodes
        self.__n_hidden_nodes = n_hidden_nodes
        self.__n_output_nodes = n_output_nodes

        # initialize weights and a bias
        if beta_init:
            if beta_init.shape != (self.__n_hidden_nodes, self.__n_output_nodes):
                raise ValueError(
                    'the shape of beta_init is expected to be %s.' % (self.__n_hidden_nodes, self.__n_output_nodes)
                )
            self.__beta = beta_init
        else:
            self.__beta = np.random.uniform(0.,1.,size=(self.__n_hidden_nodes, self.__n_output_nodes))
        if alpha_init:
            if alpha_init.shape != (self.__n_input_nodes, self.__n_hidden_nodes):
                raise ValueError(
                    'the shape of alpha_init is expected to be %s.' % (self.__n_hidden_nodes, self.__n_output_nodes)
                )
            self.__alpha = alpha_init
        else:
            self.__alpha = np.random.uniform(0.,1.,size=(self.__n_input_nodes, self.__n_hidden_nodes))
        if bias_init:
            if bias_init.shape != (self.__n_hidden_nodes,):
                raise ValueError(
                    'the shape of bias_init is expected to be %s.' % (self.__n_hidden_nodes,)
                )
            self.__bias = bias_init
        else:
            self.__bias = np.random.uniform(0.,1.,size=(self.__n_hidden_nodes,))

        # set an activation function
        if activation == 'sigmoid':
            self.__activation = _sigmoid
        elif activation == 'identity':
            self.__activation = _identity
        else:
            raise ValueError(
                'an unknown activation function \'%s\'.' % activation
            )

        # set a loss function
        if loss == 'mean_squared_error':
            self.__loss = _mean_squared_error
        elif activation == 'mean_absolute_error':
            self.__loss = _mean_absolute_error
        else:
            raise ValueError(
                'an unknown loss function \'%s\'.' % loss
            )

    def __call__(self, x):
        if len(x) == 1:
            x = np.expand_dims(x, axis=0)
        h = self.__activation(x.dot(self.__alpha) + self.__bias)
        return h.dot(self.__beta)

    def predict(self, x):
        return self(x)

    def fit(self, x, t):
        if len(x) == 1:
            x = np.expand_dims(x, axis=0)
        if len(t) == 1:
            t = np.expand_dims(t, axis=0)
        H = self.__activation(x.dot(self.__alpha) + self.__bias)

        # compute a pseudoinverse of H
        H_pinv = np.linalg.pinv(H)

        # update beta
        self.__beta = H_pinv.dot(t)

    @property
    def input_shape(self):
        return (self.__n_input_nodes,)

    @property
    def output_shape(self):
        return (self.__n_output_nodes,)

    @property
    def n_input_nodes(self):
        return self.__n_input_nodes

    @property
    def n_hidden_nodes(self):
        return self.__n_hidden_nodes

    @property
    def n_output_nodes(self):
        return self.__n_output_nodes

    @property
    def activation(self):
        if self.__activation == _sigmoid:
            return 'sigmoid'
        elif self.__activation == _identity:
            return 'identity'

    @property
    def loss(self):
        if self.__loss == _mean_squared_error:
            return 'mean_squared_error'
        elif self.__loss == _mean_absolute_error:
            return 'mean_absolute_error'
