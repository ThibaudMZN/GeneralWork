# -*- coding: utf-8 -*-

import numpy as np

class Perceptron(object):
    
    def __init__(self,nInputs=2,activation="Sign"):
        self.w = np.random.rand(nInputs)*2-1 #Init n Weights between -1 and 1
        self.activation = activation
        self.lr = 0.1
        self.b = 1
        self.meanErr = np.array(list())

    def _activation(self,x):
        if(self.activation == "Sign"):
            return np.sign(x)
        elif(self.activation == "Sigmoid"):
            return 1/(1+np.exp(x))

    def predict(self,inp):
        sumWeighted = np.dot(inp,self.w) + self.b
        return(self._activation(sumWeighted))

    def train(self,inp,labels):
        tmp = np.zeros(len(labels))
        for i in range(inp.shape[0]):
            pred = self.predict(inp[i,:])
            err = labels[i] - pred
            tmp[i] = np.abs(err)
            for j in range(self.w.shape[0]):
                self.w[j] += self.lr*err*inp[i,j]
            self.b += self.lr*err
        self.lr /= 1.5
        self.meanErr = np.hstack([self.meanErr,np.abs(np.average(tmp))])

    def __repr__(self):
        return "Perceptron Object with weights : " + str(self.w)
    

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    
    p = Perceptron(2)
    
    #input for training should be (nPoints,SizeInput)
    train = np.random.rand(100,2)
    labels = np.array(train[:,0] >= train[:,1]).astype(float)

    #plot the training data
##    red = train[labels.ravel() == 0]
##    blue = train[labels.ravel() == 1]    
##    plt.scatter(red[:,0],red[:,1],80,'r','^')
##    plt.scatter(blue[:,0],blue[:,1],80,'b','s')
##    plt.plot([0,1],[0,1],"g")

    for i in range(1000):
        p.train(train,labels)

    #plot the estimated line from perceptron
##    y0 = -(p.b/p.w[1])-(p.w[0]/p.w[1])*0
##    y1 = -(p.b/p.w[1])-(p.w[0]/p.w[1])*1
##    plt.plot([0,1],[y0,y1],"k")
##
##    plt.axis("equal")
##    plt.axis([0,1,0,1])
        
    print("Final Error : ", p.meanErr[-1])
    plt.plot(p.meanErr)
    plt.show()


    
    
