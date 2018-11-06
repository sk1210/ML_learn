from sklearn import datasets
from sklearn.utils.extmath import softmax
import numpy as np
from matplotlib import pyplot
from sklearn.preprocessing import OneHotEncoder
from activation import *
import random
import pickle

class Network:

    def __init__(self):
        self.layers = []
        self.input = None

        self.output = None

        self.x = None
        self.y = None

        self.y_hot = None
        self.classes = None

        self.delta_ouput = None

        self.lrate = 0.01
        self.batch_size = None

    def forward(self):
        input = self.input

        for i,l in enumerate(self.layers):
            #print ("Layer " , i)
            input = l.forward_prop(input)
            #print (input.shape , input)
        net.output = self.layers[-1].h_output

    def backward(self):

        error = (net.y_hot - net.output)
        delta = error * sigmoid(self.layers[-1].h_output, deriv=True)
        delta = error
        delta = np.array([np.sum(error,axis=0)/len(self.input)])
        for i in range(len(self.layers)-1,-1,-1):
            #print (i)
            self.layers[i].backward_prop(delta)
            delta = self.layers[i].back_x

    def update(self):
        # sum delta
        prev_output = net.input
        for i in range(0,len(self.layers)):
            layer = self.layers[i]

            layer.dWeight +=  self.lrate * prev_output.T.dot(layer.delta)
            #layer.weights += layer.dWeight #* layer.output
            prev_output = layer.h_output

    def updateBatch(self):

        # sum delta only

        for i in range(0,len(self.layers)):
            layer = self.layers[i]

            #self.layer.dWeight +=  self.lrate * layer.delta
            layer.weights += layer.dWeight / self.batch_size
            layer.dWeight *= 0

    def calculate_loss(self,type):
        net.y_hot = self.one_hot(self.classes,self.y)

        if type == "sse":
            loss = ((net.output - net.y_hot) ** 2)/2
            avg_loss = np.sum(loss)/len(y)

        self.delta_ouput = loss
        return loss,avg_loss

    def one_hot(self, n, x):
        encod = []
        for i in x:
            temp = [0] * n
            temp[i] = 1
            encod.append(temp)
        return np.array(encod)

    def predict(self):

        net.forward()

        y_= np.argmax(net.output)

        return y_

    def saveModel(self,name):
        f = open("pnet.weights","wb")

        weights = []
        for layer in self.layers:
            weights.append(layer.weights)

        pickle.dump(weights,f)

        f.close()

    def loadModel(self, name):
        f = open("pnet.weights", "rb")

        weights = pickle.load(f)
        for layer,weights in zip(self.layers,weights):
            layer.weights = weights

        f.close()

        print ("Model Loaded")


class Layer:
    def __init__(self, input_shape, output_shape, activation ):
        self.weights =  2*np.random.rand(input_shape,output_shape)-1
        self.bias = np.random.rand(output_shape)


        self.dWeight = np.zeros((input_shape,output_shape))
        self.dBias = None

        self.activation = activation
        self.shape = output_shape

        self.delta = None

        self.output = None
        self.h_output = None

    def forward_prop(self, x ):

        self.output = np.dot(x, self.weights)/1.0

        if self.activation == "relu":
            self.h_output = relu(self.output)

        elif self.activation == "sigmoid":
            self.h_output = sigmoid(self.output)

        elif self.activation == "tanh":
            self.h_output = tanh(self.output)

        return self.h_output

    def backward_prop(self,delta_plus):

        self.back_x = np.dot(delta_plus, self.weights.T)

        if self.activation == "sigmoid":
            self.delta = delta_plus * sigmoid(self.output, deriv=True)

        elif self.activation == "tanh":
            self.delta = delta_plus * tanh(self.output, deriv=True)

        elif self.activation == "relu":
            self.delta = delta_plus * relu(self.output, deriv=True)
        return self.back_x

    def update_weight(self):
        pass


data = datasets.load_digits()

batch_size= 1
x = data.data[:batch_size]
x = x/15
y = data.target[:batch_size]

input = x.T

layer1 = Layer(x.shape[1],64,"sigmoid")
layer2 = Layer(layer1.shape,56,"sigmoid")
layer3 = Layer(layer2.shape,64,"sigmoid")
layer4 = Layer(layer3.shape,10,"sigmoid")

net = Network()
net.classes = 10
net.layers += [layer1,layer2,layer3,layer4]

net.batch_size = 8

# Train
for i in range(10000):
    batch_size = 8
    n = random.randint(1 + batch_size, 1797 - batch_size)
    # n= 4
    x = data.data[n - 1:n + batch_size - 1] / 15
    y = data.target[n - 1:n + batch_size - 1]

    overall_loss = 0
    for j in range(batch_size):

        #print (j)
        net.input = x[j:j+1]
        net.y = y[j:j+1]

        net.forward()
        _,loss = net.calculate_loss("sse")

        overall_loss += loss
        net.backward()
        net.update()
    print (i,"Loss",overall_loss/batch_size)
    net.updateBatch()

    print( y[j], np.argmax(sigmoid(layer4.output)))

#save model
net.saveModel("model.data")

# Calc Accuracy ---------------------
net.loadModel("model.data")
correct = 0
total = 0
for i in range(len(data.data)-1):
    x = data.data[i: i+1] / 15
    y = data.target[i: i+1]
    net.input = x

    y_ = net.predict()

    if ( y[0] == y_):
        correct += 1
    else:
        print (y,y_)
    total += 1

print ( "Correct/Total", correct,total,correct/total )
print ("My Net")

# predict
