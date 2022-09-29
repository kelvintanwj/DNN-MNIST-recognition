import numpy as np

np.random.seed(1024)
from ._base_network import _baseNetwork


class TwoLayerNet(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10, hidden_size=128):
        super().__init__(input_size, num_classes)

        self.hidden_size = hidden_size
        self._weight_init()

    def _weight_init(self):
        """
        initialize weights of the network
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the first layer of shape (num_features, hidden_size)
        - b1: The bias term of the first layer of shape (hidden_size,)
        - W2: The weight matrix of the second layer of shape (hidden_size, num_classes)
        - b2: The bias term of the second layer of shape (num_classes,)
        """

        # initialize weights
        self.weights['b1'] = np.zeros(self.hidden_size)
        self.weights['b2'] = np.zeros(self.num_classes)
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.hidden_size)
        np.random.seed(1024)
        self.weights['W2'] = 0.001 * np.random.randn(self.hidden_size, self.num_classes)

        # initialize gradients to zeros
        self.gradients['W1'] = np.zeros((self.input_size, self.hidden_size))
        self.gradients['b1'] = np.zeros(self.hidden_size)
        self.gradients['W2'] = np.zeros((self.hidden_size, self.num_classes))
        self.gradients['b2'] = np.zeros(self.num_classes)

    def forward(self, X, y, mode='train'):
        """
        The forward pass of the two-layer net. The activation function used in between the two layers is sigmoid, which
        is to be implemented in self.,sigmoid.
        The method forward should compute the loss of input batch X and gradients of each weights.
        Further, it should also compute the accuracy of given batch. The loss and
        accuracy are returned by the method and gradients are stored in self.gradients

        :param X: a batch of images (N, input_size)
        :param y: labels of images in the batch (N,)
        :param mode: if mode is training, compute and update gradients;else, just return the loss and accuracy
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
            self.gradients: gradients are not explicitly returned but rather updated in the class member self.gradients
        """
        loss = None
        accuracy = None
        
        # create new matrices to encapsulate bias into the weights
        X_new = np.hstack([X, np.ones((X.shape[0],1))])
        w1_new = np.vstack([self.weights['W1'], self.weights['b1'].T])

        z1 = np.dot(X_new, w1_new) # first fully connected layer
        a1 = self.sigmoid(z1) # applying sigmoid activation

        w2 = np.vstack([self.weights['W2'], self.weights['b2'].T])
        a1_new = np.hstack([a1, np.ones((a1.shape[0],1))])

        z2 = np.dot(a1_new, w2) # second fully connected layer
        x_prob = np.array(self.softmax(z2)) # converts to probability

        loss = self.cross_entropy_loss(x_prob, y)
        accuracy = self.compute_accuracy(x_prob, y)
        
        # backwards prop
        n_class = w2.shape[1]
        n_label = y.shape[0]
        y_one_hot = np.zeros((n_label, n_class))
        y_one_hot[range(n_label), y] = 1

        # calculating W2 gradients
        dl_dz2 = (x_prob - y_one_hot)/n_label
        dz2_dw2 = np.dot(a1_new.T, dl_dz2)
        gradient2 = dz2_dw2
        self.gradients['W2'] = gradient2[:-1]

        # calculating b2 gradients
        dl_db2 = np.sum(dl_dz2, axis=0)
        self.gradients['b2'] = dl_db2

        # calculating W1 gradients
        da_dz1 = self.sigmoid_dev(z1)
        dl_da = np.dot(dl_dz2, w2[:-1].T)
        dl_dz1 = dl_da * da_dz1

        dl_dw1 = np.dot(X_new.T, dl_dz1)
        gradient1 = dl_dw1[:-1]
        self.gradients['W1'] = gradient1
        
        # calculating b1 gradients
        dl_db1 = np.sum(dl_dz1, axis=0)
        self.gradients['b1'] = dl_db1

        return loss, accuracy
