import numpy as np


class Perceptron:
    def __init__(self, N, alpha=0.1):
        self.W = np.random.randn(N+1)/np.sqrt(N) #weight initialization 
        self.alpha = alpha
    def step(self, x):
        return 1 if x > 0 else 0
    def fit(self, X, y, epochs=10):  #training happens 
        X = np.c_[X, np.ones((X.shape[0]))]

        for e in range(epochs):
            for (x, target) in zip(X, y):
                p = self.step(np.dot(x, self.W))
                if  p != target:
                    error = p - target
                    self.W += -self.alpha * error * x
    def predict(self, X, addBias = True):
        X = np.atleast_2d(X)
        if addBias:
            X = np.c_[X, np.ones((X.shape[0]))]

        return self.step(np.dot(X, self.W))

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([0, 0, 0, 1])  #bit-wise AND


print("traning perceptron...")
p = Perceptron(X.shape[1])
p.fit(X, y, epochs=20)

print("testing perceptron...")

for (x, target) in zip(X, y):
    pred = p.predict(x)
    print("{}, true label={}, pred={}".format(x, target, p.predict(x)))
    #predict on the traning data

    
    
            
        


    
