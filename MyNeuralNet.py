import numpy as np
import pandas as pd
from mnist import MNIST
import math
class AkashNet():
    def __init__(self):
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []
        self.weights = []
        self.biases = []
        self.layers = 0
        self.neurons = []
    def get_train_data(self):
        data = MNIST('C:\\Akash\\Coding\\Python\\Neural_Net\\data')
        self.x_train,self.y_train = data.load_training()
        images = []
        for data in self.x_train:
            img = np.array(data,dtype=np.uint8)
            images.append(img)
        self.x_train = np.array(images)
    def neuron_calc(self):
        while self.layers not in ['1','2']:
            self.layers = input('Please enter whether you want 1 or 2 hidden layers for your Neural Net- ')
        self.layers = int(self.layers)
        l1 = int(input('Please enter number of datapoint of input. For example, a picture with 28x28 grid, has 784 input parameters- '))
        lL = int(input('Please enter last layer of neurons. For example, a regression model should have only one layer while a classification model should have the number of class as number of neurons- '))
        hidden_neurons = (2/3)*l1+lL
        hidden_neurons = int(hidden_neurons)
        if self.layers == 1:
            self.neurons = [l1,hidden_neurons,lL]
        else:
            self.neurons = [l1,(2*hidden_neurons)//3,hidden_neurons//3,lL]
        for i in range(0,len(self.neurons)):
            if i == 0:
                print('First layer : '+str(self.neurons[i]))
            elif i == len(self.neurons)-1:
                print('Last Layer : '+ str(self.neurons[i]))
            else:
                print('Hidden Layer '+str(i)+' : '+str(self.neurons[i]))
    def random_weights_biases(self):
        for i in range(0,len(self.neurons)-1):
            self.weights.append(np.random.rand(self.neurons[i],self.neurons[i+1]))
            self.biases.append(np.full((self.neurons[i+1]),0.001))
        for weight in self.weights:
            print(weight.shape)
        for bias in self.biases:
            print(bias.shape)
    def forward_prop(self,data):
        a = data
        for layer in range(0,len(self.weights)):
            z = np.dot(a,self.weights[layer])
            z = np.add(self.biases[layer],z)
            print(z)
            a = self.softmax(z)
            print(a)
    def softmax(self,x):
        x = np.divide(x,np.max(x))
        x = np.exp(x)
        total = np.sum(x)
        x = np.divide(x,total)
        return x

        

akash = AkashNet()
akash.get_train_data()
akash.neuron_calc()
akash.random_weights_biases()
akash.forward_prop(akash.x_train[0])
        

