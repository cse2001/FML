import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl


class NaiveBayes:
    def fit(self, X, y):
        """
        X : np.array of shape (n,2)
        y : np.array of shape (n,)
        Create a variable to store number of unique classes in the dataset.
        Assume Prior for each class to be ratio of number of data points in that class to total number of data points.
        Fit a distribution for each feature for each class.
        Store the parameters of the distribution in suitable data structure, for example you could create a class for each distribution and store the parameters in the class object.
        You can create a separate function for fitting each distribution in its and call it here.
        """

        # Number of unique classes
        self.classes = np.unique(y)

        # Prior probabilities
        self.priors = {}
        for c in self.classes:
            self.priors[c] = np.sum(y == c) / len(y)

        # Fit distributions
        self.params = {}
        for c in self.classes:
            self.params[c] = {}
            X_c = X[y == c]

            # Gaussian
            self.params[c]['gaussian'] = []
            for i in range(2):
                mu = np.mean(X_c[:, i])
                sigma = np.std(X_c[:, i])
                self.params[c]['gaussian'].append((mu, sigma))

            # Bernoulli
            self.params[c]['bernoulli'] = []
            for i in range(2, 4):
                p = np.mean(X_c[:, i])
                self.params[c]['bernoulli'].append(p)

            # Laplace
            self.params[c]['laplace'] = []
            for i in range(4, 6):
                mu = np.median(X_c[:, i])
                b = np.mean(np.abs(X_c[:, i] - mu))
                self.params[c]['laplace'].append((mu, b))

            # Exponential
            self.params[c]['exponential'] = []
            for i in range(6, 8):
                lambd = 1 / np.mean(X_c[:, i])
                self.params[c]['exponential'].append(lambd)

            # Multinomial
            self.params[c]['multinomial'] = []
            k = int(np.max(X) + 1)
            alpha = 1
            for i in range(8, 10):
                counts = np.bincount(X_c[:, i].astype(int), minlength=k) + alpha
                p = counts / np.sum(counts)
                self.params[c]['multinomial'].append(p)

    def predict(self, X):
        """
        X : np.array of shape (n,2)

        Calculate the posterior probability using the parameters of the distribution calculated in fit function.
        Take care of underflow errors suitably (Hint: Take log of probabilities)
        Return an np.array() of predictions where predictions[i] is the predicted class for ith data point in X.
        It is implied that prediction[i] is the class that maximizes posterior probability for ith data point in X.
        You can create a separate function for calculating posterior probability and call it here.
        """

        def gaussian_pdf(x, mu, sigma):
            return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

        def bernoulli_pmf(x, p):
            return p ** x * (1 - p) ** (1 - x)

        def laplace_pdf(x, mu, b):
            return 0.5 * np.exp(-np.abs(x - mu) / b) / b

        def exponential_pdf(x, lambd):
            return lambd * np.exp(-lambd * x)

        def multinomial_pmf(x, p):
            return p[x]

        # Calculate posterior probabilities
        posteriors = []
        for c in self.classes:
            prior = self.priors[c]
            likelihood = 0

            # Gaussian
            for i in range(2):
                mu, sigma = self.params[c]['gaussian'][i]
                likelihood += np.log(gaussian_pdf(X[:, i], mu, sigma))

            # Bernoulli
            for i in range(2, 4):
                p = self.params[c]['bernoulli'][i - 2]
                likelihood += np.log(bernoulli_pmf(X[:, i], p))

            # Laplace
            for i in range(4, 6):
                mu, b = self.params[c]['laplace'][i - 4]
                likelihood += np.log(laplace_pdf(X[:, i], mu, b))

            # Exponential
            for i in range(6, 8):
                lambd = self.params[c]['exponential'][i - 6]
                likelihood += np.log(exponential_pdf(X[:, i], lambd))

            # Multinomial
            for i in range(8, 10):
                p = self.params[c]['multinomial'][i - 8]
                likelihood += np.log(multinomial_pmf(X[:, i].astype(int), p))

            posteriors.append(prior + likelihood)

        posteriors = np.array(posteriors).T
        predictions = np.argmax(posteriors, axis=1)

        return predictions

    def getParams(self):
        """
        Return your calculated priors and parameters for all the classes in the form of dictionary that will be used for evaluation
        Please don't change the dictionary names
        Here is what the output would look like:
        priors = {"0":0.2,"1":0.3,"2":0.5}
        gaussian = {"0":[mean_x1,mean_x2,var_x1,var_x2],"1":[mean_x1,mean_x2,var_x1,var_x2],"2":[mean_x1,mean_x2,var_x1,var_x2]}
        bernoulli = {"0":[p_x3,p_x4],"1":[p_x3,p_x4],"2":[p_x3,p_x4]}
        laplace = {"0":[mu_x5,mu_x6,b_x5,b_x6],"1":[mu_x5,mu_x6,b_x5,b_x6],"2":[mu_x5,mu_x6,b_x5,b_x6]}
        exponential = {"0":[lambda_x7,lambda_x8],"1":[lambda_x7,lambda_x8],"2":[lambda_x7,lambda_x8]}
        multinomial = {"0":[[p0_x9,...,p4_x9],[p0_x10,...,p7_x10]],"1":[[p0_x9,...,p4_x9],[p0_x10,...,p7_x10]],"2":[[p0_x9,...,p4_x9],[p0_x10,...,p7_x10]]}
        """

        priors = self.priors
        gaussian = {}
        bernoulli = {}
        laplace = {}
        exponential = {}
        multinomial = {}

        for c in self.classes:
            # Gaussian
            gaussian[c] = []
            mu1, sigma1 = self.params[c]['gaussian'][0]
            mu2, sigma2 = self.params[c]['gaussian'][1]
            gaussian[c].extend([mu1, mu2])
            gaussian[c].extend([sigma1 ** 2, sigma2 ** 2])

            # Bernoulli
            bernoulli[c] = self.params[c]['bernoulli']

            # Laplace
            laplace[c] = []
            mu5, b5 = self.params[c]['laplace'][0]
            mu6, b6 = self.params[c]['laplace'][1]
            laplace[c].extend([mu5, mu6])
            laplace[c].extend([b5, b6])

            # Exponential
            exponential[c] = self.params[c]['exponential']

            # Multinomial
            multinomial[c] = self.params[c]['multinomial']

        return (priors, gaussian, bernoulli, laplace, exponential, multinomial)


def save_model(model, filename="model.pkl"):
    """

    You are not required to modify this part of the code.

    """
    file = open("model.pkl", "wb")
    pkl.dump(model, file)
    file.close()


def load_model(filename="model.pkl"):
    """

    You are not required to modify this part of the code.

    """
    file = open(filename, "rb")
    model = pkl.load(file)
    file.close()
    return model


def visualise(data_points, labels, filename):
    """
    datapoints: np.array of shape (n,2)
    labels: np.array of shape (n,)
    filename: str, name of the file to save the plot
    """

    plt.figure(figsize=(8, 6))
    plt.scatter(data_points[:, 0], data_points[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title('Generated 2D Data from 5 Gaussian Distributions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig(filename)
    plt.show()


def net_f1score(predictions, true_labels):
    """Calculate the multclass f1 score of the predictions.
    For this, we calculate the f1-score for each class

    Args:
        predictions (np.array): The predicted labels.
        true_labels (np.array): The true labels.

    Returns:
        float(list): The f1 score of the predictions for each class
    """

    def precision(predictions, true_labels, label):
        """Calculate the multclass precision of the predictions.
        For this, we take the class with given label as the positive class and the rest as the negative class.

        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The precision of the predictions.
        """
        """Start of your code."""

        tp = np.sum((predictions == label) & (true_labels == label))
        fp = np.sum((predictions == label) & (true_labels != label))

        if tp + fp == 0:
            return 0

        return tp / (tp + fp)

    def recall(predictions, true_labels, label):
        """Calculate the multclass recall of the predictions.
        For this, we take the class with given label as the positive class and the rest as the negative class.
        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The recall of the predictions.
        """
        """Start of your code."""
        tp = np.sum((predictions == label) & (true_labels == label))
        fn = np.sum((predictions != label) & (true_labels == label))

        if tp + fn == 0:
            return 0

        return tp / (tp + fn)

    def f1score(predictions, true_labels, label):
        """Calculate the f1 score using it's relation with precision and recall.

        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The f1 score of the predictions.
        """

        """Start of your code."""
        p = precision(predictions, true_labels, label)
        r = recall(predictions, true_labels, label)

        if p + r == 0:
            return 0

        f1 = 2 * p * r / (p + r)
        """End of your code."""
        return f1

    f1s = []
    for label in np.unique(true_labels):
        f1s.append(f1score(predictions, true_labels, label))
    return f1s


def accuracy(predictions, true_labels):
    """

    You are not required to modify this part of the code.

    """
    return np.sum(predictions == true_labels) / predictions.size


if __name__ == "__main__":
    """

    You are not required to modify this part of the code.

    """

    # Load the data
    train_dataset = pd.read_csv('./data/train_dataset.csv', index_col=0).to_numpy()
    validation_dataset = pd.read_csv('./data/validation_dataset.csv', index_col=0).to_numpy()

    # Extract the data
    train_datapoints = train_dataset[:, :-1]
    train_labels = train_dataset[:, -1]
    validation_datapoints = validation_dataset[:, 0:-1]
    validation_labels = validation_dataset[:, -1]

    # Visualize the data
    visualise(train_datapoints, train_labels, "train_data.png")

    # Train the model
    model = NaiveBayes()
    model.fit(train_datapoints, train_labels)

    # Make predictions
    train_predictions = model.predict(train_datapoints)
    validation_predictions = model.predict(validation_datapoints)

    # Calculate the accuracy
    train_accuracy = accuracy(train_predictions, train_labels)
    validation_accuracy = accuracy(validation_predictions, validation_labels)

    # Calculate the f1 score
    train_f1score = net_f1score(train_predictions, train_labels)
    validation_f1score = net_f1score(validation_predictions, validation_labels)

    # Print the results
    print('Training Accuracy: ', train_accuracy)
    print('Validation Accuracy: ', validation_accuracy)
    print('Training F1 Score: ', train_f1score)
    print('Validation F1 Score: ', validation_f1score)

    # Save the model
    save_model(model)

    # Visualize the predictions
    visualise(validation_datapoints, validation_predictions, "validation_predictions.png")
