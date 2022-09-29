import numpy as np

from ._base_network import _baseNetwork

class SoftmaxRegression(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10):
        """
        A single layer softmax regression. The network is composed by:
        a linear layer without bias => (activation) => Softmax
        :param input_size: the input dimension
        :param num_classes: the number of classes in total
        """
        super().__init__(input_size, num_classes)
        self._weight_init()

    def _weight_init(self):
        '''
        initialize weights of the single layer regression network. No bias term included.
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the linear layer of shape (num_features, hidden_size)
        '''
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.num_classes)
        self.gradients['W1'] = np.zeros((self.input_size, self.num_classes))

    def forward(self, X, y, mode='train'):
        """
        Compute loss and gradients using softmax with vectorization.

        :param X: a batch of image (N, 28x28)
        :param y: labels of images in the batch (N,)
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
        """
        loss = None
        gradient = None
        accuracy = None
        """
        Steps:
        1) Implement the forward process and compute the Cross-Entropy loss
        2) Compute the gradient of the loss with respect to the weights
        """
        z = np.dot(X, self.weights['W1']) # fully connected layer
        a = self.ReLU(z) # applying ReLU activation
        x_prob = self.softmax(a) # converts to probability

        loss = self.cross_entropy_loss(x_prob, y)
        accuracy = self.compute_accuracy(x_prob, y)
        gradient = X.T.dot(loss)/y.shape[0]
        if mode != 'train':
            return loss, accuracy

        """
        Implement the backward process:
        1) Compute gradients of each weight by chain rule
        2) Store the gradients in self.gradients
        """
        W = self.weights['W1']
        n_class = W.shape[1]
        n_label = y.shape[0]
        y_one_hot = np.zeros((n_label, n_class))
        y_one_hot[range(n_label), y] = 1

        # calculating the intermediate steps
        dl_da = (x_prob - y_one_hot)/n_label
        da_dz = self.ReLU_dev(a)
        dz_dw = X.T

        # finally, apply chain rule
        dl_dz = dl_da * da_dz
        dl_dw = np.dot(dz_dw, dl_dz)

        # store gradients in self.gradients
        gradient = dl_dw
        self.gradients['W1'] = gradient
        return loss, accuracy

        