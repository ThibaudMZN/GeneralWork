### -*- coding: utf-8 -*-

import numpy as np

class Perceptron(object):
    
    def __init__(self,nInputs=2,activation="Sign"):
        self.w = np.random.rand(nInputs)*2-1 #Init n+1 (for bias) Weights between -1 and 1
        self.activation = activation
        self.lr = 0.01
        self.b = 1
        self.meanErr = np.array(list())

    def _activation(self,x):
        if(self.activation == "Sign"):
            return float(np.sign(x))
        elif(self.activation == "Sigmoid"):
            return float(1/(1+np.exp(-x)))

    def predict(self,inp):
        sumWeighted = np.dot(inp,self.w) + self.b
        return(self._activation(sumWeighted))

    def train(self,inp,labels):#,n_epoch):
        #for k in range(n_epoch):
        tmp = np.zeros(len(labels))
        for i in range(inp.shape[0]):
            pred = self.predict(inp[i,:])
            err = labels[i] - pred
            tmp[i] = np.abs(err)
            self.w += self.lr*err*inp[i,:]
            self.b += self.lr*err
        self.meanErr = np.hstack([self.meanErr,np.average(tmp)])

    def __repr__(self):
        return "Perceptron Object with weights : " + str(self.w)
    

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    
    p = Perceptron(2)
    
    #input for training should be (nPoints,SizeInput)
    train = np.random.rand(100,2)
    labels = np.array(train[:,0] >= train[:,1]).astype(float)*2-1

    #plot the training data
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    red = train[labels.ravel() == -1]
    blue = train[labels.ravel() == 1]    
    ax.scatter(red[:,0],red[:,1],80,'r','^')
    ax.scatter(blue[:,0],blue[:,1],80,'b','s')
    line1, = ax.plot([0,1],[0,0],"k")
    ax.plot([0,1],[0,1],"g:")
    plt.axis("equal")
    plt.axis([0,1,0,1])
    
    for i in range(100):
        p.train(train,labels)#,500)

    #plot the estimated line from perceptron
        y0 = -(p.b/p.w[1])-(p.w[0]/p.w[1])*0
        y1 = -(p.b/p.w[1])-(p.w[0]/p.w[1])*1
        line1.set_ydata([y0,y1])
        fig.canvas.draw()

        
    print("Final Error : ", p.meanErr[-1])
    






















