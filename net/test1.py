from sklearn import datasets
import numpy as np
from matplotlib import pyplot
from sklearn.preprocessing import OneHotEncoder

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

        self.lrate = 0.5

    def forward(self):
        input = self.input

        for i,l in enumerate(self.layers):
            #print ("Layer " , i)
            input = l.forward_prop(input)
            #print (input.shape , input)
        net.output = input

    def backward(self):

        error = net.output - net.y_hot
        delta = -error * sigmoid(self.layers[-1].output, deriv=True)

        for i in range(len(self.layers)-1,1,-1):
            print (i)
            self.layers[i].backward_prop(delta)
            delta = self.layers[i].delta

    def update(self):

        for i in range(len(self.layers)):
            layer = self.layers[i]

            layer.weights += self.lrate * layer.delta * layer.output


    def calculate_loss(self,type):
        net.y_hot = self.one_hot(self.classes,y)

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

class Layer:
    def __init__(self, input_shape, output_shape, activation ):
        self.weights =  np.random.rand(input_shape,output_shape)
        self.bias = np.random.rand(output_shape)

        self.dWeight = np.random.rand(output_shape,input_shape)
        self.dBias = None

        self.activation = activation
        self.shape = output_shape

        self.delta = None

        self.output = None
        self.h_output = None

    def forward_prop(self, x ):

        self.output = np.dot(x, self.weights)

        if self.activation == "relu":
            self.h_output = np.max(0,self.output)

        elif self.activation == "sigmoid":
            self.h_output = 1/(1+np.exp(self.output))

        return self.h_output

    def backward_prop(self,delta_plus):

        back_out = np.dot(delta_plus ,self.weights.T )

        if self.activation == "sigmoid":
            self.delta = back_out * sigmoid(self.output, deriv=True)

        return self.delta

    def update_weight(self):
        pass

def sigmoid(x,deriv = False):
    if deriv:
        return x*(1-x)

data = datasets.load_digits()
x = data.data[:2]
y = data.target[:2]

input = x.T

layer1 = Layer(x.shape[1],5,"sigmoid")
layer2 = Layer(layer1.shape,7,"sigmoid")
layer3 = Layer(layer2.shape,9,"sigmoid")
layer4 = Layer(layer3.shape,10,"sigmoid")


net = Network()
net.classes = 10
net.layers += [layer1,layer2,layer3,layer4]
net.input = x
net.y = y


# forward prop
for i in range(1):
    net.forward()
    loss=  net.calculate_loss("sse")
    print (i,loss[1])
    net.backward()
    #net.update()

print("My Net")
