import numpy as np

class LogisticRegression:
    def __init__(self):
        """
        Initialize `self.weights` properly.
        Recall that for binary classification we only need 1 set of weights (hence `num_classes=1`).
        We have given the default zero intialization with bias term (hence the `d+1`).
        You are free to experiment with various other initializations including random initialization.
        Make sure to mention your initialization strategy in your report for this task.
        """
        self.num_classes = 1  # single set of weights needed
        self.d = 2  # input space is 2D. easier to visualize
        self.weights = np.zeros((self.d + 1, self.num_classes))
        self.v = np.zeros_like(self.weights)
    def preprocess(self, input_x):
        """
        Preprocess the input any way you seem fit.
        """

        # Normalizing all the d features of X.
        for _ in range(self.d):
            input_x = (input_x - input_x.mean(axis=0)) / input_x.std(axis=0)
        return input_x

    def sigmoid(self, x):
        """
        Implement a sigmoid function if you need it. Ignore otherwise.
        """
        return 1.0 / (1 + np.exp(-x))

    def calculate_loss(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: a single scalar value corresponding to the loss.
        """
        N,dummy = input_x.shape
        Z = np.dot(input_x, self.weights[:-1]) + self.weights[-1][0]
        P = self.sigmoid(Z)
        input_y = input_y.reshape(-1, 1)
        loss = (-1 / N) * np.sum(input_y * np.log(P) + (1 - input_y) * np.log(1 - P))
        return loss

    def calculate_gradient(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: the gradient of loss function wrt weights.
        Ensure that gradient.shape == self.weights.shape.
        """
        N= input_x.shape[0]
        Z = np.dot(input_x, self.weights[:-1]) + self.weights[-1][0]
        P = self.sigmoid(Z)
        gradient = np.zeros((self.d + 1, self.num_classes))
        input_y=input_y.reshape(-1,1)
        gradient[:-1] = (1 / N) * np.dot(input_x.T, (P - input_y))
        gradient[-1][0] = np.sum(P - input_y)
        return gradient

    def update_weights(self, grad, learning_rate, momentum):
        """
        Arguments:
        grad -- NumPy array with same shape as `self.weights`
        learning_rate -- scalar
        momentum -- scalar
        Returns: nothing
        The function should update `self.weights` with the help of `grad`, `learning_rate` and `momentum`
        """

        self.v[:-1] = momentum * self.v[:-1] - learning_rate * grad[:-1]
        self.weights[:-1] = self.weights[:-1] + self.v[:-1]
        self.v[-1][0] = momentum * self.v[-1][0] - learning_rate * grad[-1][0]
        self.weights[-1][0] = self.weights[-1][0] + self.v[-1][0]



    def get_prediction(self, input_x):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        Returns: a NumPy array with shape (N,)
        The returned array must be the list of predicted class labels for every input in `input_x`
        """
        z = np.dot(input_x, self.weights[:-1]) + self.weights[-1][0]
        y_pred = []
        for i in self.sigmoid(z):
            if i>=0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)
        return y_pred


class LinearClassifier:
    def __init__(self):
        self.num_classes = 3
        self.d = 4
        self.weights = np.zeros((self.d+1, self.num_classes))
        self.v=np.zeros_like(self.weights)

    def preprocess(self, train_x):
        return np.concatenate([train_x, np.ones((train_x.shape[0], 1))], axis=1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def calculate_loss(self, input_x, input_y):
        N = input_x.shape[0]
        Z = np.dot(input_x, self.weights)
        P = self.sigmoid(Z)
        y_oh = np.eye(self.num_classes)[input_y]
        loss = (-1 / N) * np.sum(y_oh * np.log(P) + (1 - y_oh) * np.log(1 - P))
        return loss


    def calculate_gradient(self, input_x, input_y):
        N = input_x.shape[0]
        Z = np.dot(input_x, self.weights)
        P = self.sigmoid(Z)
        y_oh = np.eye(self.num_classes)[input_y]
        gradient = (-1 / N) * np.dot(input_x.T, (y_oh - P))
        return gradient

    def update_weights(self, grad, learning_rate, momentum):
        self.v =momentum * self.v - learning_rate * grad
        self.weights = self.weights + self.v

    def get_prediction(self, input_x):
        z = np.dot(input_x, self.weights)
        sigs=self.sigmoid(z)
        y_pred = np.argmax(sigs,axis=1)
        return y_pred
