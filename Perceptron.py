import numpy as np

class Perceptron(object):
    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activation(summation)

    def activation(self, summation):
        return 1 if summation > 0 else 0
        # if summation > 0:
        #     return 1
        # else:
        #     return 0

    def train(self, training_inputs, training_outputs):
        for _ in range(self.threshold):
            for inputs, output in zip(training_inputs, training_outputs):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (output - prediction) * inputs
                self.weights[0] += self.learning_rate * (output - prediction)
