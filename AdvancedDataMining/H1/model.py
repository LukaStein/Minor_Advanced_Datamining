class Perceptron():
        def __init__(self, dim):
              self.dim = dim
              self.bias = 0
              self.weights = [0, 0]

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
                x1 = xCoords[0]
                x2 = xCoords[1]
                # initiële voorspelling
                yValue = self.bias + (self.weights[0] * x1) + (self.weights[1] * x2)
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
                  #   print(x, y, yhat[index])
                  # update-regel
                  self.bias = self.bias - (yhat[index] - yOld)
                  self.weights[0] = self.weights[0] - (yhat[index] - yOld) * x[0]
                  self.weights[1] = self.weights[1] - (yhat[index] - yOld) * x[1]
                  # opnieuw voorspellen
                  yNew = self.bias + (self.weights[0] * x[0]) + (self.weights[1] * x[1])
                  predictLabel = lambda x : -1.0 if x < 0  else (0.0 if x == 0 else 1.0)
                  yhatNew = predictLabel(yNew)
                  yhatNewPredictions.append(yhatNew)
                  index += 1
             return yhatNewPredictions
            #  print(yhatNewPredictions)
                # #print(sgn, biasUpdate, weight1, x1, weight2, x2)
        
        def fit(self, xs, ys, *, epochs=0):
            #  print(self.partial_fit(xs[:5], ys[:5]))
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
          self.weights = [0, 0]

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
               x1 = xCoords[0]
               x2 = xCoords[1]
               # initiële voorspelling
               yValue = self.bias + (self.weights[0] * x1) + (self.weights[1] * x2)
               # bewaar voorspellingen 
               predictions_yhats.append(yValue)
          return predictions_yhats
     
     def partial_fit(self, xs, ys, *, alpha=0.01):
          yhat : list = self.predict(xs)
          index = 0
          yhatNewPredictions = []
          for x,yOld in zip(xs,ys):
               #   print(x, y, yhat[index])
               # update-regel
               self.bias = self.bias - alpha*(yhat[index] - yOld)
               self.weights[0] = self.weights[0] - alpha*(yhat[index] - yOld) * x[0]
               self.weights[1] = self.weights[1] - alpha*(yhat[index] - yOld) * x[1]
               # opnieuw voorspellen
               yNew = self.bias + (self.weights[0] * x[0]) + (self.weights[1] * x[1])
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


def linear(a):
     return a

def sign(yValue):
     predictLabel = lambda x : -1.0 if x < 0  else (0.0 if x == 0 else 1.0)
     return predictLabel(yValue)

from math import e

def tanh(yValue):
     return (e**yValue - e**-yValue) / (e**yValue + e**-yValue)

def sigmoid(yValue):
     return (1 / (1+e**-yValue))

def mean_squared_error(yhat, y):
     # print("mean", y)
     return (yhat-y)**2

def mean_absolute_error(yhat, y):
     return abs(yhat-y)


def derivative(function, delta=0.1):
     
     def wrapper_derivative(x, *args):
          # print("wrapper", args) # probleem is dat y niet geupdate wordt en dus op 0 blijft
          # print(function(x + delta*x, *args))
          return  ( function(x + delta, *args) - function(x - delta, *args) )  / (2*delta)

          wrapper_derivative.__name__ = function.__name__ + "'"
          wrapper_derivative.__qualname__ = function.__qualname__ + "'"
     return wrapper_derivative


class Neuron():
     def __init__(self, dim, activation=linear, loss=mean_squared_error):
          self.dim = dim
          self.bias = 0
          self.weights = [0,0]
          self.activation = activation
          self.loss = loss

     def __repr__(self):
          text = f'Neuron(dim={self.dim}, activation={self.activation.__name__}, loss={self.loss.__name__})'
          return text

     def predict(self, xs) -> list: 
          predictions_yhat = []
          for xCoords in xs:
               x1 = xCoords[0]
               x2 = xCoords[1]
               yValue = self.bias + (self.weights[0] * x1) + (self.weights[1] * x2)
               yhat = self.activation(yValue)
               predictions_yhat.append(yhat)
          return predictions_yhat

     def partial_fit(self, xs, ys, *, alpha=0.01) -> list:
          yhat : list = self.predict(xs)
          index = 0 # aantal voorspelde labels moet gelijk zijn aan aantal echte labels
          predictions_updated_yhat = []
          dl_dhat = derivative(self.loss) # mean_squared_error per default, geef functie aan derivative
          dyhat_da = derivative(self.activation)
          for xCoords, yOld in zip(xs,ys): # yOld is echte label
               # print(dl_dhat(yhat[index], yOld))
               self.bias = self.bias - ( alpha * dl_dhat(yhat[index], yOld) * dyhat_da(alpha) )
               self.weights[0] = self.weights[0] - ( alpha * dl_dhat(yhat[index], yOld) * dyhat_da(alpha) * xCoords[0])
               self.weights[1] = self.weights[1] - ( alpha * dl_dhat(yhat[index], yOld) * dyhat_da(alpha) * xCoords[1])
               # voorspellingen opnieuw doen
               yNew = self.bias + (self.weights[0] * xCoords[0]) + (self.weights[1] * xCoords[1])
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