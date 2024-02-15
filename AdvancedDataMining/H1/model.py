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
             for x,yOld in zip(xs,ys):
                #   print(x, y, yhat[index])
                  # update-regel
                  self.bias = self.bias - (yhat[index] - yOld)
                  self.weights[0] = self.weights[0] - (yhat[index] - yOld) * x[0]
                  self.weights[1] = self.weights[1] - (yhat[index] - yOld) * x[1]
                  # opnieuw voorspellen
                  yNew = self.bias + (self.weights[0] * x[0]) + (self.weights[1] * x[1])
                  predictLabel = lambda x : -1.0 if x < 0  else (0.0 if x == 0 else 1.0)
                  predictLabel(yNew)
                  index += 1
            #  print(yhatNewPredictions)
                # #print(sgn, biasUpdate, weight1, x1, weight2, x2)
            