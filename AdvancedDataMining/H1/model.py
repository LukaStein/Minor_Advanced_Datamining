from collections import Counter
from copy import deepcopy
from random import uniform

class Perceptron():
        def __init__(self, dim):
              self.dim = dim
              self.bias = 0
              self.weights = [0 for _ in range(dim)]

        def __repr__(self):
            text = f'Perceptron(dim={self.dim})'
            return text
        
        def predict(self, xs) -> list:
            """
            xs ontvangt attributen van een lijst instances
            param: geneste lijst van lijsten
            return: enkelvoudige lijst
            """
            predictions_yhat = [] 
            for xCoords in xs:
                # initiële voorspelling
                yValue = self.bias + sum(self.weights[xi] * xCoords[xi] for xi in range(len(xCoords)))
                predictLabel = lambda x : -1.0 if x < 0  else (0.0 if x == 0 else 1.0)
                yhat = predictLabel(yValue)
                # bewaar voorspellingen 
                predictions_yhat.append(yhat)
            return predictions_yhat
        
        def partial_fit(self, xs, ys):
             yhat : list = self.predict(xs)
             index = 0
             yhatNewPredictions = []
             for x,yOld in zip(xs,ys):
                  # update-regel
                  self.bias = self.bias - (yhat[index] - yOld)
                  for xi in range(len(x)):
                       self.weights[xi] = self.weights[xi] - (yhat[index] - yOld) * x[xi]
                  # opnieuw voorspellen
                  yNew = self.bias + sum(self.weights[xi] * x[xi] for xi in range(len(x)))
                  predictLabel = lambda x : -1.0 if x < 0  else (0.0 if x == 0 else 1.0)
                  yhatNew = predictLabel(yNew)
                  yhatNewPredictions.append(yhatNew)
                  index += 1
             return yhatNewPredictions
        
        def fit(self, xs, ys, *, epochs=0):
             if epochs > 0: # choose number of epochs to iterate over
                  for _ in range(epochs):
                       self.partial_fit(xs, ys)
             elif epochs < 0: # not allowed epochs input
                  print("Epoch below 0 isn't allowed")
             else: # default or zero epoch input value
                  index = 0
                  for index in range(len(ys)):
                       yhatNewPredictions = self.partial_fit(xs, ys)
                       if yhatNewPredictions[index] != ys[index]:
                            self.partial_fit(xs, ys)
                            index += 1


class LinearRegression:
     def __init__(self, dim):
          self.dim = dim
          self.bias = 0
          self.weights = [0 for _ in range(dim)]

     def __repr__(self):
          text = f'Perceptron(dim={self.dim})'
          return text
     
     def predict(self, xs) -> list:
          """
          xs ontvangt attributen van een lijst instances
          param: geneste lijst van lijsten
          return: enkelvoudige lijst
          """
          predictions_yhats = [] 
          for xCoords in xs:
               # print(self.weights)
               yValue = 0
               for xi in range(len(xCoords)):
                    # print(self.weights[xi])
                    yValue = yValue + self.weights[xi] * xCoords[xi]
               # initiële voorspelling
               yValue = yValue + self.bias
               # bewaar voorspellingen 
               predictions_yhats.append(yValue)
          return predictions_yhats
     
     def partial_fit(self, xs, ys, *, alpha=0.01):
          yhat : list = self.predict(xs)
          index = 0
          yhatNewPredictions = []
          for x,yOld in zip(xs,ys):
               # update-regel
               self.bias = self.bias - alpha*(yhat[index] - yOld)
               for xi in range(len(x)):
                    self.weights[xi] = self.weights[xi] - alpha*(yhat[index] - yOld) * x[xi]
               # opnieuw voorspellen
               yNew = self.bias + sum(self.weights[xi] * x[xi] for xi in range(len(x)))
               yhatNewPredictions.append(yNew)
               index += 1
          return yhatNewPredictions
          #  print(yhatNewPredictions)
               # #print(sgn, biasUpdate, weight1, x1, weight2, x2)
     
     def fit(self, xs, ys, *, alpha=0.001, epochs=100):
          #  print(self.partial_fit(xs[:5], ys[:5]))
          # self.partial_fit(xs, ys, alpha=0.001)
          if epochs > 0: # choose number of epochs to iterate over
               for _ in range(epochs):
                    self.partial_fit(xs, ys, alpha=alpha)
          elif epochs <= 0: # not allowed epochs input
               print("Epoch below 0 isn't allowed")


def linear(a): # activatiefunctie
     return a

def sign(yValue): # activatiefunctie
     predictLabel = lambda x : -1.0 if x < 0  else (0.0 if x == 0 else 1.0)
     return predictLabel(yValue)

from math import e

def tanh(yValue): # activatiefunctie
     return (e**yValue - e**-yValue) / (e**yValue + e**-yValue)

def sigmoid(yValue): # activatiefunctie
     return (1 / (1+e**-yValue))

def softsign(yValue): # activatiefunctie
     return yValue / (1 + abs(yValue))

def mean_squared_error(yhat, y): # loss
     # print("mean", y)
     return (yhat-y)**2

def mean_absolute_error(yhat, y): # loss
     return abs(yhat-y)

def hinge(yhat, y): # loss
     return max(1-yhat*y,0)


def derivative(function, delta=0.8):
     
     def wrapper_derivative(x, *args):
          # print("wrapper", args) # probleem is dat y niet geupdate wordt en dus op 0 blijft
          # print(function(x + delta*x, *args))
          wrapper_derivative.__name__ = function.__name__ + "'"
          wrapper_derivative.__qualname__ = function.__qualname__ + "'"
          return  ( function(x + delta, *args) - function(x - delta, *args) )  / (2*delta)

         
     return wrapper_derivative


class Neuron():
     def __init__(self, dim, activation=linear, loss=mean_squared_error):
          self.dim = dim
          self.bias = 0
          self.weights = [0 for _ in range(dim)]
          self.activation = activation
          self.loss = loss

     def __repr__(self):
          text = f'Neuron(dim={self.dim}, activation={self.activation.__name__}, loss={self.loss.__name__})'
          return text

     def predict(self, xs) -> list:
          """
          xs ontvangt attributen van een lijst instances
          param: geneste lijst van lijsten
          return: enkelvoudige lijst
          """
          predictions_yhats = [] 
          for xCoords in xs:
               yValue = 0
               for xi in range(len(xCoords)):
                    # print(self.weights[xi])
                    yValue = yValue + self.weights[xi] * xCoords[xi]
               # initiële voorspelling
               yValue = yValue + self.bias
               # bewaar voorspellingen 
               predictions_yhats.append(yValue)
          return predictions_yhats

#Predictions: TypeAlias = list[float] / Typing module
     def partial_fit(self, xs, ys, *, alpha=0.01) -> list: # dit kan ooklist[list[float]]
          yhat : list = self.predict(xs)
          index = 0 # aantal voorspelde labels moet gelijk zijn aan aantal echte labels
          predictions_updated_yhat = []
          dl_dhat = derivative(self.loss) # mean_squared_error per default, geef functie aan derivative
          dyhat_da = derivative(self.activation)
          for xCoords, yOld in zip(xs,ys): # yOld is echte label
               self.bias = self.bias - ( alpha * dl_dhat(yhat[index], yOld) * dyhat_da(alpha) )
               for xi in range(len(xCoords)):
                    self.weights[xi] = self.weights[xi] - (alpha * dl_dhat(yhat[index], yOld) * dyhat_da(alpha) * xCoords[xi])
               # voorspellingen opnieuw doen
               yNew = self.bias + sum(self.weights[xi] * xCoords[xi] for xi in range(len(xCoords)))

               yhatUpdate = self.activation(yNew)
               predictions_updated_yhat.append(yhatUpdate)
               index += 1
          return predictions_updated_yhat
 
     def fit(self, xs, ys, *, alpha=0.001, epochs=100):
          #  print(self.partial_fit(xs[:5], ys[:5]))
          # self.partial_fit(xs, ys, alpha=0.001)
          if epochs > 0: # choose number of epochs to iterate over
               for _ in range(epochs):
                    self.partial_fit(xs, ys, alpha=alpha)
          elif epochs <= 0: # not allowed epochs input
               print("Epoch below 0 isn't allowed")




class Layer():
     layercounter = Counter()
     
     def __init__(self, outputs, *, name=None, next=None):
          Layer.layercounter[type(self)] += 1
          if name is None:
               name = f'{type(self).__name__}_{Layer.layercounter[type(self)]}'
          self.inputs = 0
          self.outputs = outputs
          self.name = name
          self.next = next
          
     def __repr__(self):
          text = f'Layer(inputs={self.inputs}, outputs={self.outputs}, name={repr(self.name)})'
          if self.next is not None:
               text += ' + ' + repr(self.next)
          return text

     def __add__(self, next):
          result = deepcopy(self)
          result.add(deepcopy(next))
          return result
     
     def __getitem__(self, index):
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
          if self.validate_type_and_dimension(other):
               components = (x + y for x, y in zip(self.components, other.components))
               self._components = tuple(components)
               return self
          raise NotImplemented
     
     def __len__(self):
          return len(self)
     
     def __iter__(self):
        return iter(self)
     
     def __call__(self, xs, loss=None, ys=None):
        print(xs, loss, ys)
        raise NotImplementedError('Abstract __call__ method')

     def add(self, next):
          if self.next is None:
               self.next = next
               next.set_inputs(self.outputs) # stuur output van deze layer naar layer volgende laag als input
          else:
               self.next.add(next)

     def set_inputs(self, inputs):
          self.inputs = inputs

class InputLayer(Layer): # dus hier geef je de begin data door

     def __repr__(self):
          text = f'InputLayer(outputs={self.outputs}, name={repr(self.name)})'
          if self.next is not None:
               text += ' + ' + repr(self.next)
          return text
     
     def set_inputs(self):
          raise(NotImplementedError)
     
     def __call__(self, xs, ys=None, alpha=None):
          return self.next(xs, ys=ys, alpha=alpha) # input naar Denselayer

     def predict(self, xs):
          yhats, ls, gs = self(xs)
          return yhats # eind voorspellingen naar de vorige layer
     
     def evaluate(self, xs, ys):
          yhats, ls, gs = self(xs, ys=ys) # de call van DenseLayer
          l_mean = sum(ls) / len(ls)
          return l_mean

     def partial_fit(self, xs, ys, alpha=0.001):
          self(xs, ys=ys, alpha=alpha)

     def fit(self, xs, ys, *, alpha=0.001, epochs=100):
          if epochs > 0: # choose number of epochs to iterate over
               for _ in range(epochs):
                    self.partial_fit(xs, ys=ys, alpha=alpha)
          elif epochs <= 0: # not allowed epochs input
               print("Epoch below 0 isn't allowed")

class DenseLayer(Layer):
     def __init__(self, outputs, name=None, next=None):
          self.name = name
          super().__init__(outputs, name=name,next=next)
          self.bias = [0.0 for _ in range(0, self.outputs)]  # aantal biases gelijk aan aantal neurons
          self.weights = None
     
     def __repr__(self):
          text = f'DenseLayer(outputs={self.outputs}, name={repr(self.name)})'
          if self.next is not None:
               text += ' + ' + repr(self.next)
          return text
     
     def set_inputs(self, inputs):
          # print(inputs)
          if inputs is not None:
               self.inputs = inputs

               border : float = (6/(inputs+self.outputs))**(1/2) 
               self.weights = [[uniform(-border,border) for i in range(inputs)] for _ in range(self.outputs)] # aantal weights (i) in een aantal neurons (o) 

     def __call__(self, xs, ys=None, alpha=None):
          # print(xs)
          # print(ys)
          # print(alpha)
          aa = []   # Uitvoerwaarden voor alle instances xs
          # print(self.weights)
          for n in range(len(xs)): # instances
               a = []   # Uitvoerwaarde voor één instance x
               for o in range(self.outputs): # neuronen
                    # Bereken voor elk neuron o met de lijst invoerwaarden x de uitvoerwaarde
                    preactivation = self.bias[o]
                    for i in range(self.inputs):
                         
                         preactivation += self.weights[o][i] * xs[n][i]
                    a.append(preactivation)
               aa.append(a)
          yhats, ls, gs = self.next(aa, ys=ys, alpha=alpha) # van de activationlayer = naar de activationlayer
          # print(gs)
          # print("dense", self.next) # waar gaat de data heen
          # print("dense", self.inputs)
          # bereken qs (nieuwe gradiënten)
          qs = None
          if alpha is not None: # als er een learning rate is, wil je een gradient descent
               qs = []
               for n in range(len(xs)): # instances
                    q = []
                    for i in range(self.inputs): # neuronen, elke output van inputlayer als input
                         gradient = sum(gs[n][o] * self.weights[o][i] for o in range(self.outputs))
                         q.append(gradient)
                    qs.append(q)   
                    # updaten gewichten en biases
                    for o in range(self.outputs):  # update weights and biases for each output
                         update = alpha / len(xs) * gs[n][o]
                         self.bias[o] -= update  # update bias
                         for i in range(self.inputs):  # update weights
                              self.weights[o][i] -= update * xs[n][i]
          return yhats, ls, qs # naar inputlayer, maar wordt daar niet gebruikt
     
class ActivationLayer(Layer):
     def __init__(self, outputs, activation=linear, name=None, next=None):
          self.activation = activation
          self.name = name
          self.activation_gradient = derivative(self.activation)

          super().__init__(outputs, name=name, next=next)

     def __repr__(self):
          text = f'ActivationLayer(outputs={self.outputs}, name={repr(self.name)}, activation={self.activation.__name__})'
          if self.next is not None:
               text += ' + ' + repr(self.next)
          return text
     
     def __call__(self, xs, ys=None, alpha=None):

          hh = []   # Uitvoerwaarden voor alle instances xs
          for x in xs: # eigenlijk a in aa, instances
               h = []   # Uitvoerwaarde voor één instance x
               for o in range(self.outputs): # neuronen
                    # Bereken voor elk neuron o met de lijst invoerwaarden x de uitvoerwaarde
                    h_no = self.activation(x[o])
                    h.append(h_no)
               hh.append(h)
               # hh.append([self.activation(x[o]) for o in range(self.outputs)])
          yhats, ls, gs = self.next(hh, ys=ys, alpha=alpha) # van de denselayer = naar de losslayer
          # print("activation", self.next) # waar gaat de data heen 
          # print("activation", self.inputs)
          qs = None # gradient descent
          if alpha is not None:
               qs = []
               for n in range(len(xs)): # itereer over instances
                    q = []
                    for o in range(self.inputs): # itereer over neuronen, inputs van outputs vorige laag
                         gradient = gs[n][o] * self.activation_gradient(xs[n][o])
                         q.append(gradient)
                    qs.append(q)
          return yhats, ls, qs # naar denselayer

class LossLayer(Layer):
     def __init__(self, loss=mean_squared_error, name=None):
          self.loss = loss
          self.name = name
          self.loss_gradient = derivative(self.loss)

     def __repr__(self):
          text = f'LossLayer(name={repr(self.name)}, loss={self.loss.__name__})'
          return text
     
     def add(self):
          raise(NotImplementedError)
     
     def __call__(self, xs, ys=None, alpha=None) -> list:
          # print("loss",self.inputs)
          yhats = xs
          ls, gs = None, None
          if ys is not None:
               ls = []
               for n in range(len(xs)): # instances
                    loss = 0.0
                    for i in range(self.inputs):
                         loss += self.loss(yhats[n][i], ys[n][i])
                    ls.append(loss)
               if alpha is not None:
                    gs = []
                    for n in range(len(xs)):
                         g = []
                         for i in range(self.inputs):
                              gradient = self.loss_gradient(yhats[n][i], ys[n][i])
                              g.append(gradient)
                         gs.append(g)
          return yhats, ls, gs # naar activationlayer