# IMPORTS
from collections import Counter
from copy import deepcopy
from math import e, log  # natural value and log
from random import uniform, seed, shuffle, Random


class Perceptron:
    """
    A kind of model (calculation unit) of what a neuron does in the brain.
    """

    def __init__(self, dim):
        """Construct a Perceptron
        :param dim: dimensions of an instance
        """
        self.dim = dim
        self.bias = 0
        self.weights = [0 for _ in range(dim)]

    def __repr__(self) -> str:
        """
        Return a string representation of the Perceptron
        """
        text = f'Perceptron(dim={self.dim})'
        return text

    def predict(self, xs) -> list:
        """
        Predict classifier in a single class problem
        :param self:
        :param xs; nested list of lists. XS receives attributes of a list instances
        :return: predictions_yhat; single list of yhat predictions
        """
        predictions_yhat = []
        for xCoords in xs:
            # Initial prediction
            y_value = self.bias + sum(self.weights[xi] * xCoords[xi] for xi in range(len(xCoords)))
            predict_label = lambda x: -1.0 if x < 0 else (0.0 if x == 0 else 1.0)
            yhat = predict_label(y_value)
            # Save predictions
            predictions_yhat.append(yhat)
        return predictions_yhat

    def partial_fit(self, xs, ys):
        """
        Adjust weights and biases to partially fit/correct the predictions
        :param xs: nested list of lists. XS receives attributes of a list instances
        :param ys: list of true labels
        :return: number of updates made
        """
        converge_steps = 0
        for x, y in zip(xs, ys):
            yhat = self.bias + sum(self.weights[xi] * x[xi] for xi in range(len(x)))
            yhat = -1.0 if yhat < 0 else (0.0 if yhat == 0 else 1.0)
            if yhat != y:
                update = y - yhat
                self.bias += update
                for xi in range(len(x)):
                    self.weights[xi] += update * x[xi]
                converge_steps += 1
        return converge_steps

    def fit(self, xs, ys, *, epochs=0):
        """
        Keep invoking the partially fit function over n epochs or if epochs=0
        run infinite epochs until all predictions are correct
        :param xs: nested list of lists. XS receives attributes of a list instances
        :param ys: list of true labels
        :param epochs: number of epochs to partially fit on, default is 0 (run until convergence)
        """
        if epochs == 0:
            epoch = 0
            while True:
                converge_steps = self.partial_fit(xs, ys)
                epoch += 1
                if converge_steps == 0:
                    print(f"Converged after {epoch} epochs.")
                    break
        elif epochs > 0:
            for epoch in range(1, epochs + 1):
                converge_steps = self.partial_fit(xs, ys)
                if converge_steps == 0:
                    print(f"Converged after {epoch} epochs.")
                    break
            else:
                print(f"Did not converge within the given {epochs} epochs.")
        else:
            print(f"Can't run on negative epochs: {epochs}")


class LinearRegression:
    """
    Model for linear regression between classes
    """
    def __init__(self, dim):
        """Construct a linear regression model (LRM)
        :param dim: dimensions of an instance"""
        self.dim = dim
        self.bias = 0
        self.weights = [0 for _ in range(dim)]

    def __repr__(self) -> str:
        """
        Return a string representation of the LRM
        """
        text = f'Perceptron(dim={self.dim})'
        return text

    def predict(self, xs) -> list:
        """
        Predict classifier in a single class problem
        :param self:
        :param xs; nested list of lists. XS receives attributes of a list instances
        :return: predictions_yhat; single list of yhat predictions
        """
        predictions_yhats = []
        for xCoords in xs:
            # print(self.weights)
            y_value = 0
            for xi in range(len(xCoords)):
                # print(self.weights[xi])
                y_value = y_value + self.weights[xi] * xCoords[xi]
            # initiële voorspelling
            y_value = y_value + self.bias
            # bewaar voorspellingen
            predictions_yhats.append(y_value)
        return predictions_yhats

    def partial_fit(self, xs, ys, *, alpha=0.01) -> list:
        """
        Adjust weights and biases to partially fit/correct the predictions
        :param self:
        :param xs; nested list of lists. XS receives attributes of a list instances
        :param ys; nested list of lists. YS contains the true labels.
        :param alpha; learning rate, defaulted to 0.01
        :return: yhat_new_predictions; partially fitted predictions
        """
        yhat: list = self.predict(xs)
        index = 0
        for x, yOld in zip(xs, ys):
            # update-regel
            self.bias = self.bias - alpha * (yhat[index] - yOld)
            for xi in range(len(x)):
                self.weights[xi] = self.weights[xi] - alpha * (yhat[index] - yOld) * x[xi]
            index += 1

    def fit(self, xs, ys, *, alpha=0.001, epochs=100):
        """
        Keep invoking the partially fit function over n epochs or if n=0
        run infinite epochs until all predictions are correct
        :param self:
        :param xs; nested list of lists. XS receives attributes of a list instances
        :param ys; nested list of lists. YS contains the true labels.
        :param alpha; learning rate, defaulted to 0.001
        :param epochs; number of epochs to partially fit on, defaulted to 100
        """
        if epochs > 0:  # choose number of epochs to iterate over
            for _ in range(epochs):
                self.partial_fit(xs, ys, alpha=alpha)
        elif epochs <= 0:  # not allowed epochs input
            print("Epoch below 0 isn't allowed")


def linear(a, *args) -> tuple[float, int]:  # activation function
    """
    Identify function returns the predicted y-value that it receives
    :param a; predicted y-value
    :return: a; predicted y-value
    """
    return a


def sign(y_value, *args) -> float:  # activation function
    """
    Signum function that returns -1 for negative y-values, 0 for neutral y-values
    and 1 for positive y-values
    :param y_value: predicted y-value
    :return: -1, 0 or 1 as predicted label
    """
    predict_label = lambda x: -1.0 if x < 0 else (0.0 if x == 0 else 1.0)
    return predict_label(y_value)


def tanh(y_value, *args) -> float:  # activation function
    """
    Tangens hyperbolic function flattens the slope of signum function
    :param y_value: predicted y-value
    :return: adjusted y_value
    """
    if y_value > 700:
        return 1
    if y_value < -700:
        return -1
    else:
        return (e ** y_value - e ** -y_value) / (e ** y_value + e ** -y_value)


def sigmoid(y_value, beta=1) -> float:  # activation function
    """
    Sigmoid function that adds the natural log e in the equation.
    :param y_value: predicted y-value
    :param beta: The trainable parameter beta is constant for sigmoid.
    :return: adjusted y_value
    """
    if y_value < -700:  # handle overflow
        return e ** y_value
    return 1 / (1 + e ** -(beta * y_value))


def softsign(y_value, *args) -> float:  # activation function
    """
    Softsign function provides non-linearity
    :param y_value: predicted y-value
    :return: adjusted y_value
    """
    return y_value / (1 + abs(y_value))


def softplus(y_value, *args) -> float:  # activation function
    """
    Softplus function provides non-linearity, but can't calculate negative y-values i.e. y.min = 0.
    Softplus is the average of the relu function and provides a gradual slope.
    :param y_value: predicted y-value
    :return: adjusted y_value
    """
    if y_value > 700:   # handle overflow
        return y_value
    return log(1 + e ** y_value)


def relu(y_value, *args) -> int:  # activation function
    """
    Relu function provides linearity with a slope of 0 or 1, but can't calculate negative y-values i.e. y.min = 0.
    :param y_value: predicted y-value
    :return: adjusted y_value
    """
    return max(0, y_value)


def swish(y_value, beta=1) -> float:  # activation function
    """
    Swish function receives a trainable parameter beta and is the sigmoid function times the y_value.
     Per default, beta is 1.
    :param beta:
    :param y_value: predicted y-value
    :param beta: The trainable parameter beta can be adjusted.
    :return: adjusted y_value
    """
    return y_value * sigmoid(y_value, beta)


def nipuna(y_value, beta=1) -> float:  # activation function
    """
    The nipuna function is a newer function that compares the output of the swish function with the y_value.
    :param y_value: predicted y-value
    :param beta: The trainable parameter beta can be adjusted.
    :return: adjusted y_value
    """
    return max(swish(y_value, beta), y_value)


def mean_squared_error(yhat, y):  # loss
    """
    Calculates the mean squared error between true and predicted labels
    :param yhat: predicted y-value
    :param y: true y-value
    :return: loss value
    """
    return (yhat - y) ** 2


def mean_absolute_error(yhat, y):  # loss
    """
    Calculates the mean absolute error between true and predicted labels
    :param yhat: predicted y-value
    :param y: true y-value
    :return: loss value
    """
    return abs(yhat - y)


def hinge(yhat, y):  # loss
    """
    Calculates the hinge loss, by comparing if the loss is larger than 0.
    :param yhat: predicted y-value
    :param y: true y-value
    :return: loss value
    """
    return max(1 - yhat * y, 0)


def categorical_crossentropy(yhat_no, y_no, epsilon=0.01):  # loss
    """
    Calculates the categorical cross entropy loss for multi nominal classification problems
    :param yhat_no:
    :param y_no:
    :param epsilon:
    :return: loss value
    """
    if yhat_no >= epsilon:  # if not to close to zero
        return -y_no * log(yhat_no)
    return -y_no * (log(epsilon) + (yhat_no - epsilon) / epsilon)  # else take log of e instead of yhat_no


def binary_crossentropy(yhat_no, y_no, epsilon=0.0001):  # loss
    """
    Calculates the binary cross entropy loss for bi-nominal classification problems i.e. 0 or 1
    :param yhat_no:
    :param y_no:
    :param epsilon:
    :return: loss value
    """
    return -y_no * pseudo_log(yhat_no, epsilon) - (1 - y_no) * pseudo_log(1 - yhat_no, epsilon)  # binary cross-entropy formula


def pseudo_log(yhat_no, epsilon):
    if yhat_no >= epsilon: 
        return log(yhat_no)
    else:
        return log(epsilon) + (yhat_no - epsilon) / epsilon # default solution if yhat_no is almost 0
    

def derivative(function, delta=0.03) -> "wrapper_derivative":
    """
    Returns derivative of either an activation function or loss function
    :param function: Any activation or loss function
    :param delta: step size for delta x
    :return: derived function made with wrapper_derivative
    """
    def wrapper_derivative(x, *args) -> "derived_function":
        """
        Calculate derivative of either an activation function or loss function
        :param x: Any activation or loss function
        :param args: optional arguments
        :return: derived function of received function
        """
        wrapper_derivative.__name__ = function.__name__ + "'"
        wrapper_derivative.__qualname__ = function.__qualname__ + "'"
        return (function(x + delta, *args) - function(x - delta, *args)) / (2 * delta)

    return wrapper_derivative


class Neuron:
    """
    Cell body (neuron) is activated by impulse from a dendrite.
    Can predict and perform regression
    """
    def __init__(self, dim, activation=linear, loss=mean_squared_error):
        """
        Construct a neuron. With dimensions for an instance, activation and loss function are implemented, and
        passed here.
        :param dim: dimensions of an instance
        :param activation: activation function, per default linear
        :param loss: loss_function, per default mean_squared_error
        """
        self.dim = dim
        self.bias = 0
        self.weights = [0 for _ in range(dim)]
        self.activation = activation
        self.loss = loss

    def __repr__(self) -> str:
        """
        Return a string representation of the neuron
        """
        text = f'Neuron(dim={self.dim}, activation={self.activation.__name__}, loss={self.loss.__name__})'
        return text

    def predict(self, xs) -> list:
        """
        Predict classifier in a single class problem
        :param self:
        :param xs; nested list of lists. XS receives attributes of a list instances
        :return: predictions_yhat; single list of yhat predictions
        """
        predictions_yhats = []
        for xCoords in xs:
            y_value = 0
            for xi in range(len(xCoords)):
                y_value = y_value + self.weights[xi] * xCoords[xi] + self.bias
            # Initial prediction
            y_value = self.activation(y_value) # phi mist
            # Save predictions
            predictions_yhats.append(y_value)
        return predictions_yhats

    def partial_fit(self, xs, ys, *, alpha=0.01) -> list:
        """
        Adjust weights and biases to correct/update the predictions. Activation and loss functions
        are derived to introduce gradient descent (flattening out the deviations in accuracy and loss curve)
        :param self:
        :param xs; nested list of lists. XS receives attributes of a list instances
        :param ys; nested list of lists. YS contains the true labels.
        :param alpha; learning rate, defaulted to 0.01
        :return: predictions_updated_yhat; partially fitted predictions
        """
        yhat: list = self.predict(xs)
        index = 0  # number of predicted labels has to be equal to number of true labels
        predictions_updated_yhat = []
        dl_dhat = derivative(self.loss)  # pass the loss function to get its derivative
        dyhat_da = derivative(self.activation)  # pass the activation function to get its derivative

        for xCoords, yOld in zip(xs, ys):  # yOld is the true label
            self.bias = self.bias - (alpha * dl_dhat(yhat[index], yOld) * dyhat_da(alpha)) # update bias
            for xi in range(len(xCoords)):  
                self.weights[xi] = self.weights[xi] - (
                        alpha * dl_dhat(yhat[index], yOld) * dyhat_da(alpha) * xCoords[xi]) # update weights
            index += 1

    def fit(self, xs, ys, *, alpha=0.001, epochs=100):
        """
        Keep invoking the partially fit function over n epochs
        :param self:
        :param xs; nested list of lists. XS receives attributes of a list instances
        :param ys; nested list of lists. YS contains the true labels.
        :param alpha; learning rate, defaulted to 0.001
        :param epochs; number of epochs (full runs) to partially fit on, defaulted to 100
        """
        if epochs > 0:  # Choose number of epochs to iterate over
            for _ in range(epochs):
                self.partial_fit(xs, ys, alpha=alpha)
        elif epochs <= 0:  # Not allowed epochs input
            print("Epoch below 0 isn't allowed")


class Layer:
    """
    Parent layer holding all functionality to connect layers and send input to the next layer.
    The layer class is part of networks known as the multi-layered perceptron or neural network, that are
    capable of deep learning.
    """
    layercounter = Counter()  # Count the layers like the name implies

    def __init__(self, outputs, *, name=None, next=None):
        """
        Construct the initial/parent layer
        :param outputs: Output of current layer that will be sent as input to next layer. I.e. initial instances of xs.
        :param name: The name of the initialised layer
        :param next: Points to the next layer
        """
        Layer.layercounter[type(self)] += 1
        if name is None:
            name = f'{type(self).__name__}_{Layer.layercounter[type(self)]}'
        self.inputs = 0
        self.outputs = outputs
        self.name = name
        self.next = next

    def __repr__(self) -> str:
        """
        Return a string representation of a layer
        """
        text = f'Layer(inputs={self.inputs}, outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def __add__(self, next):
        """
        Adds a new layer by first copying the received next layer and then adding it.
        :param next: Points to c
        :return: result; contains all added layers
        """
        result = deepcopy(self)
        result.add(deepcopy(next))
        return result

    def __getitem__(self, index):
        """
        Keeps layers (counts and names) in check
        :param index: layer index
        :return: self; class instance
        """
        if index == 0 or index == self.name:
            return self
        if isinstance(index, int):
            if self.next is None:
                raise IndexError('Layer index out of range')
            return self.next[index - 1]
        if isinstance(index, str):
            if self.next is None:
                raise KeyError(index)
            return self.next[index]
        raise TypeError(f'Layer indices must be integers or strings, not {type(index).__name__}')

    def __iadd__(self, other):
        """
        Implements += add functionality for layer classes
        :param other: a layer instance
        :return: self; class instance
        """
        if self.validate_type_and_dimension(other):
            components = (x + y for x, y in zip(self.components, other.components))
            self._components = tuple(components)
            return self
        raise NotImplemented

    def __len__(self):
        """
        Implements length functionality
        :return: length of self instance
        """
        return len(self)

    def __iter__(self):
        """
        Implements iterator functionality
        :return: iterate over self instance
        """
        return iter(self)

    def __call__(self, xs, loss=None, ys=None):
        """
        Abstract call method with later purpose to call the next layer.
        :param xs; nested list of lists. XS receives attributes of a list instances
        :param loss: loss function
        :param ys; nested list of lists. YS contains the true labels.
        :return:
        """
        raise NotImplementedError('Abstract __call__ method')

    def add(self, next):
        """
        Set outputs of current layer as inputs for the next layer.
        :param next:
        :return:
        """
        if self.next is None:  # set Input layer as first layer
            self.next = next
            next.set_inputs(self.outputs)  # stuur output van deze layer naar layer volgende laag als input
        else: # set next layer after a previous layer exists
            self.next.add(next)

    def set_inputs(self, inputs):
        """
        Initialise inputs
        :param inputs: outputs set as inputs
        """
        self.inputs = inputs


class InputLayer(Layer):
    """
    Layer that receives start data
    """
    def __repr__(self) -> str:
        """
        Return a string representation of the input layer
        """
        text = f'InputLayer(outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def set_inputs(self):
        """
        Set inputs is not implemented in this class as it passes the outputs as inputs to the Dense layer. Meaning
        no input initialisation is needed here.
        """
        raise (NotImplementedError)

    def __call__(self, xs, ys=None, alpha=None):
        """
        Call next layer and pass input instances, true labels and alpha to the next layer
        :param xs; nested list of lists. XS receives attributes of a list instances
        :param ys; nested list of lists. YS contains the true labels.
        :param alpha: learning rate, defaulted to None
        :return: self.next; the next layer that is the Dense layer
        """
        return self.next(xs, ys=ys, alpha=alpha)

    def predict(self, xs):
        """
        Send the start input to the Dense layer.
        Receive the final predicted yhats and return them
        :param xs; nested list of lists. XS receives attributes of a list instances
        :return: final predicted yhats
        """
        yhats, ls, gs = self(xs)
        return yhats

    def evaluate(self, xs, ys):
        """
        Evaluate the average loss. Send xs and ys to the Dense layer and receive the loss value.
        :param xs; nested list of lists. XS receives attributes of a list instances
        :param ys; nested list of lists. YS contains the true labels.
        :return: l_mean; average loss
        """
        yhats, ls, gs = self(xs, ys=ys)  # call of dense layer
        l_mean = sum(ls) / len(ls)
        return l_mean

    def partial_fit(self, xs, ys, batch_size=None, alpha=0.001):
        """
        Initialise a batch_size and send batches to the Dense layer. The Dense layer returns predictions, loss
        and the gradients. Though only the losses are used for calculating the average loss over all batches
        :param xs; nested list of lists. XS receives attributes of a list instances
        :param ys; nested list of lists. YS contains the true labels.
        :param batch_size: size of batch (data set size), defaulted to the entire data set
        :param alpha: learning rate, defaulted to 0.001
        :return: mean of loss
        """
        if batch_size is None:
            batch_size = len(xs)

        loss_sum, loss_len = 0, 0
        for batch in range(0, len(xs), batch_size):  # from zero to length of list in steps of batch_size
            xs_batch = xs[batch:batch + batch_size]
            ys_batch = ys[batch:batch + batch_size]
            yhats, ls, gs = self(xs_batch, ys=ys_batch, alpha=alpha)  # de call van DenseLayer
            loss_sum += sum(ls)
            loss_len += len(ls)

        l_mean = loss_sum / loss_len
        return l_mean

    def fit(self, xs, ys, *, validation_data=(None, None), batch_size=None, alpha=0.001, epochs=100):
        """
        Invoke partial fitting over n epochs. Also, evaluate the mean loss between the predictions and true labels.
        Xs and ys are shuffled randomly, so that every epoch will have unique training and validation.
        :param xs; nested list of lists. XS receives attributes of a list instances
        :param ys; nested list of lists. YS contains the true labels.
        :param validation_data: Tuple of instances and true labels to test on for validation
        :param batch_size: size of batch (data set size), defaulted to the entire data set
        :param alpha: learning rate, defaulted to 0.001
        :param epochs; number of epochs (full runs) to partially fit on, defaulted to 100
        :return: history dictionary containing either, only training loss values or includes validation loss values
        """
        if epochs > 0:  # choose number of epochs to iterate over
            history = {'loss': []}
            history.update(
                {'val_loss': []} if all(validation_data) else history)  # add validation_data loss values if not empty
            for epoch in range(epochs):
                seed(1234)  # same seed
                r = Random(1234)  # same seed
                shuffle(xs)  # shuffle data
                r.shuffle(ys)  # shuffle data
                l_mean = self.partial_fit(xs=xs, ys=ys, batch_size=batch_size, alpha=alpha)
                history['loss'].append(l_mean)
                if len(history) == 2:  # two keys are present
                    xs_val, ys_val = validation_data
                    vl_mean = self.evaluate(xs_val, ys_val)
                    history['val_loss'].append(vl_mean)
            return history
        elif epochs <= 0:  # not allowed epochs input
            raise ValueError("Epoch below 0 isn't allowed")


class DenseLayer(Layer):
    """
    Layer that penalizes weights and biases when the model is making mistakes. It introduces gradient descent, i.e.
    slowly slope down to the correct values to predict. Also, pre-activation values are calculated.
    """
    def __init__(self, outputs, name=None, next=None):
        """
        Construct the Dense layer
        :param outputs: Outputs of previous layer that will be sent as input to next layer. Initially instances of xs.
        :param name: The name of the initialised layer
        :param next: Points to the next layer
        """
        self.name = name
        super().__init__(outputs, name=name, next=next)  # send outputs, name and next layer info to parent layer
        self.bias = [0.0 for _ in range(0, self.outputs)]  # number of baises equal to neurons
        self.weights = None


    def __repr__(self) -> str:
        """
        Return a string representation of the Dense layer
        """
        text = f'DenseLayer(outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def set_inputs(self, inputs):
        """
        If inputs are received from the input layer, set random weights for each input over all neurons
        :param inputs: xs; nested list of lists. XS receives attributes of a list instances
        """
        if inputs is not None:  # check if inputs are received from input layer
            self.inputs = inputs

            border: float = (6 / (inputs + self.outputs)) ** (1 / 2)  # set range value to randomize in between
            self.weights = [[uniform(-border, border) for i in range(inputs)] for _ in
                            range(self.outputs)]  # number of weights (i) in number of neurons (o)

    def __call__(self, xs, ys=None, alpha=None):
        """
        Call next Activation (or Softmax activation) layer and send pre-activation values, true labels and alpha.
        Receive from the next layer the predicted y_hats, loss values and gradient descent values.
        Pre-activation values are calculated, biases and weights are updated.
        :param xs; nested list of lists. XS receives attributes of a list instances
        :param ys; nested list of lists. YS contains the true labels.
        :param alpha: learning rate, defaulted to None
        :return: yhats, ls, qs; predictions, losses and gradient descent values
        """
        aa = []  # Output values for all instances in xs
        for n in range(len(xs)):  # Instances
            a = []  # Output value for one instance x
            for o in range(self.outputs):  # Neurons
                # Calculate the output (pre-activation) value for each neuron o with the list of input values x
                preactivation = self.bias[o]
                for i in range(self.inputs):
                    preactivation += self.weights[o][i] * xs[n][i]
                a.append(preactivation)
            aa.append(a)
        yhats, ls, gs = self.next(aa, ys=ys, alpha=alpha)  # from activation layer <=> to activation layer
        # Calculate qs (new gradients)
        qs = None
        if alpha is not None:  # If there is a learning rate, you want a gradient descent
            qs = []
            for n in range(len(xs)):  # Instances
                q = []
                for i in range(self.inputs):  # Neurons, each output of input layer as input
                    gradient = sum(gs[n][o] * self.weights[o][i] for o in range(self.outputs))
                    q.append(gradient)
                qs.append(q)
                # Update weights and biases for each output and input
                for o in range(self.outputs):
                    update = alpha / len(xs) * gs[n][o]
                    self.bias[o] -= update  # Update bias
                    for i in range(self.inputs):
                        self.weights[o][i] -= update * xs[n][i] # Update weights
        return yhats, ls, qs  # To input layer


class ActivationLayer(Layer):
    """
    General output layer. Calculates predictions with an activation function.
    """
    def __init__(self, outputs, activation=linear, beta=1, name=None, next=None):
        """
        Construct the Activation layer
        :param outputs: Pre-activation values of previous Dense layer that will be used to calculate output values
        :param activation; Activation function to be passed, defaulted linear
        :param beta; If an activation function requires a trainable beta parameter, defaulted to 1, i.e. constant
        :param name: The name of the initialised layer
        :param next: Points to the next layer
        """
        self.activation = activation
        self.name = name
        self.activation_gradient = derivative(self.activation)
        self.beta = beta

        super().__init__(outputs, name=name, next=next)  # send outputs, name and next layer info to parent layer

    def __repr__(self) -> str:
        """
        Return a string representation of the Activation layer
        """
        text = f'ActivationLayer(outputs={self.outputs}, name={repr(self.name)}, activation={self.activation.__name__})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def __call__(self, xs, ys=None, alpha=None) -> "yhats, ls, qs":
        """
        Call next loss layer and send output (predictions), true labels and alpha. The loss layer returns
        the same predictions to this layer, the calculated losses and the gradient descent values.
        New gradient descent values are calculated with the derived activation function.
        :param xs; Pre-activation values of previous Dense layer that will be used to calculate output values
        :param ys; nested list of lists. YS contains the true labels.
        :param alpha: learning rate, defaulted to None
        :return: yhats, ls, qs; predictions, losses and gradient descent values
        """
        hh = []  # aa : Output values for all instances of xs
        for x in xs:  # Each instance (list) in the list of all instances
            h = []  # Output value for one instance x
            for o in range(self.outputs):  # Neurons
                # Calculate the output value for each neuron o with the list of input values x.
                h_no = self.activation(x[o], self.beta)
                h.append(h_no)
            hh.append(h)
        yhats, ls, gs = self.next(hh, ys=ys, alpha=alpha)  # from Dense layer <=> to the Loss layer
        qs = None  # Gradient descent
        if alpha is not None:  # If there is a learning rate, you want a gradient descent
            qs = []
            for n in range(len(xs)):  # Iterate over instances
                q = []
                for o in range(self.inputs):  # Iterate over neurons o, inputs of outputs from previous layer
                    gradient = gs[n][o] * self.activation_gradient(xs[n][o])
                    q.append(gradient)
                qs.append(q)
        return yhats, ls, qs  # To dense layer


class SoftmaxLayer(Layer):  # uitvoerlaag neurale netwerken
    """
    Output layer of neural networks. Calculates predictions with softmax activation function.
    """
    def __init__(self, outputs, name=None, next=None):
        """
        Construct the Softmax activation layer
        :param outputs: Pre-activation values of previous Dense layer that will be used to calculate output values
        :param name: The name of the initialised layer
        :param next: Points to the next layer
        """
        super().__init__(outputs, name=name, next=next) # send outputs, name and next layer info to parent layer

    def __repr__(self) -> str:
        """
        Return a string representation of the SoftMax activation layer
        """
        text = f'ActivationLayer(outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def __call__(self, xs, ys=None, alpha=None) -> "yhats, ls, gs":
        """
        Call next loss layer and send output (predictions), true labels and alpha. The loss layer returns
        the same predictions to this layer, the calculated losses and the gradient descent values.
        :param xs; Pre-activation values of previous Dense layer that will be used to calculate output values
        :param ys; nested list of lists. YS contains the true labels.
        :param alpha: learning rate, defaulted to None
        :return: yhats, ls, qs; predictions, losses and gradient descent values
        """
        yhat_noses = []  # hh : list with for each instance a list of yhat predictions of each output
        for n in range(len(xs)):
            yhat_nose = []  # List with predictions of an instance
            max_val = max(xs[n])
            for o in range(self.outputs):  # i == o for n inputs == outputs
                xs[n][o] = xs[n][o] - max_val

                yhat_no = e ** xs[n][o]
                yhat_nose.append(yhat_no)  # Fill list with denominators
            yhat_sum = sum(yhat_nose)  # Calculate sum of all outputs for given instance
            yhat_nose = [i / yhat_sum for i in yhat_nose]  # Where i = e^x_no (yhat_nose get overwritten with outputs)
            yhat_noses.append(yhat_nose)

        yhats, ls, gs = self.next(yhat_noses, ys=ys, alpha=alpha)  # from Dense layer <=> to the Loss layer

        qs = None  # Gradient descent
        if alpha is not None:
            qs = []
            for n in range(len(xs)):  # Iterate over instances
                q = []
                for i in range(self.inputs):  # Iterate over neurons o, inputs of outputs from previous layer
                    # Calculate gradient descent value
                    gradient = sum(gs[n][o] * yhats[n][o] * ((o == i) - yhats[n][i]) for o in range(self.outputs))
                    q.append(gradient)
                qs.append(q)
        return yhats, ls, qs  # To dense layer


class LossLayer(Layer):
    """
    Evaluation loss layer. Calculates losses and loss gradient values- (-with the derived loss function).
    Predictions received from the activation layer are used,
    but not altered i.e. are returned identical to the activation layer.
    This is the last layer! There is no next.
    """
    def __init__(self, loss=mean_squared_error, name=None):
        """
        Construct loss layer.
        :param loss: Function to calculate loss, defaulted to mean_squared_error.
        The loss value shows the closer to zero the better is the prediction accuracy.
        :param name: The name of the initialised layer.
        """
        self.loss = loss
        self.name = name
        self.loss_gradient = derivative(self.loss)

    def __repr__(self) -> str:
        """
        Return a string representation of the loss layer
        """
        text = f'LossLayer(name={repr(self.name)}, loss={self.loss.__name__})'
        return text

    def add(self):
        """
        Since there is no next layer, functionality to add disappears.
        :return:
        """
        raise (NotImplementedError)

    def __call__(self, xs, ys=None, alpha=None) -> "yhat, ls, gs":
        """
        No call to next layer. But loss values and gradients descent loss values are calculated.
        :param xs; Activation values of previous activation layer that will be used to calculate loss values
        :param ys; nested list of lists. YS contains the true labels.
        :param alpha: learning rate, defaulted to None
        :return: yhats, ls, gs; predictions, losses and gradient descent values
        """
        yhats = xs
        ls, gs = None, None
        if ys is not None:
            ls = []
            for n in range(len(xs)):  # Instances
                loss = 0.0
                for i in range(self.inputs):
                    loss += self.loss(yhats[n][i], ys[n][i]) # calculate loss values
                ls.append(loss)
            if alpha is not None:
                gs = []
                for n in range(len(xs)):
                    g = []
                    for i in range(self.inputs):
                        gradient = self.loss_gradient(yhats[n][i], ys[n][i]) # calculate gradient descent loss values
                        g.append(gradient)
                    gs.append(g)
        return yhats, ls, gs  # To activation layer
