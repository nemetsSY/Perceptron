

import numpy as np
from Perceptron import Perceptron

training_inputs = []
training_inputs.append(np.array([1,1,0]))
training_inputs.append(np.array([1,0,0]))
training_inputs.append(np.array([1,0,-1]))
training_inputs.append(np.array([0,0.5,1]))
training_inputs.append(np.array([0,5,3]))

outputs = np.array([1, 0, 1, 1.5, -3])

perceptron = Perceptron(3)
perceptron.train(training_inputs, outputs)

print(perceptron.weights)

inputs = np.array([1, 1, 1])
print(perceptron.predict(inputs))

inputs = np.array([1, -1, 0])
print(perceptron.predict(inputs))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
