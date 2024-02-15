class Perceptron():
        def __init__(self, dim):
              self.dim = dim
              self.bias = 0
              self.weights = 0

        def __repr__(self):
            text = f'Perceptron(dim={self.dim})'
            return text
        
        def predict(self, xs) -> list:
            """
            xs ontvangt attributen van een lijst instances
            param: geneste lijst van lijsten
            return: enkelvoudige lijst
            """
            predictions = [] 
            for xCoords in xs:
                x1 = xCoords[0]
                x2 = xCoords[1]
                # initiële voorspelling
                sgn = self.bias + (self.weights * x1) + (self.weights * x2)
                predict = lambda x : -1.0 if x < 0  else (0.0 if x == 0 else 1.0)
                yhat = predict(sgn)
                # bewaar voorspellingen 
                predictions.append(yhat)
            return predictions
        
        def partial_fit(self, xs, ys):
             yhat = self.predict(xs)
             print(yhat)
             ys_yhat = [ys, yhat]
             print(ys_yhat)
             for x,y in zip(xs,ys):
                  print(x, y)
                  # update-regel
                # biasUpdate = self.bias - (yhat - sgn)
                # weight1 = self.weights - (yhat - sgn) * x1
                # weight2 = self.weights - (yhat - sgn) * x2
                # # opnieuw voorspellen
                # #print(sgn, biasUpdate, weight1, x1, weight2, x2)
                # sgnUpdate = biasUpdate + (weight1 * x1) + (weight2 * x2)
                # yhatTraining = predict(sgnUpdate)
            