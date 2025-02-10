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
        self.error_rates = []
        self.outputs = []
        self.z =[]
        self.final_output = []
        self.weight_error = []
        self.bias_error = []
        self.learning_rate = 0.01
    def get_train_data(self):
        data = MNIST('.\\data')
        self.x_train,self.y_train = data.load_training()
        images = []
        for data in self.x_train:
            img = np.array(data,dtype=np.uint8)
            images.append(img)
        self.x_train = np.array(images)/255.0
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
            self.weight_error.append(np.full((self.neurons[i],self.neurons[i+1]),0))
            self.biases.append(np.full((self.neurons[i+1]),0.001))
            self.bias_error.append(np.full((self.neurons[i+1],1),0))
        for weight in self.weights:
            print(weight.shape)
        for bias in self.biases:
            print(bias.shape)
    def forward_prop(self,index):
        a = self.x_train[index]
        self.outputs.append(a)
        self.z.append(a)
        for layer in range(0,len(self.weights)):
            z = np.dot(a,self.weights[layer])
            z = np.add(self.biases[layer],z)
            print(z.shape)
            print('z')
            self.z.append(z)
            if layer == len(self.weights)-1:
                a = self.softmax(z)
                print(a)
                print('a')
            else:
                
                a = self.leaky_relu(z)
                print(a)
                print('a')
            self.outputs.append(a)
        self.final_output = a
        print(self.final_output)
        print('final_output')
    def softmax(self,x):
        print(x)
        x = x-np.max(x)
        print('x')
        x = np.exp(x)
        total = np.sum(x)
        x = np.divide(x,total)
        return x
    def find_error_mat(self,index):
        i = self.y_train[index]
        arr = np.full((1,10),0)
        arr[0][i] = 1
        return arr
    def find_error_rate(self,index):
        y_target = self.find_error_mat(index)
        y_target = np.subtract(self.final_output,y_target)
        print(y_target)
        print('y_target')

        for i in range(0,len(self.neurons)-1):
            self.error_rates.append([])
        self.error_rates.append(y_target)
        print(self.error_rates)
        print('error rates')

  
    def find_derva_relu(self,x):
        x = np.where(x>10**100,0,x)
        x = np.where(np.logical_and(x>0 ,x<10**100),1,x)
        x = np.where(x<0,0.01,x)
        x = np.where(x<-10**100,0,x)
        return x
    def leaky_relu(self,x):
        x = np.where(x>10**100,10**100,x)
        x = np.where(x<0,0.01*x,x)
        x = np.where(x<-10**100,-10**100,x)
        return x

    def backpropogate(self):
        for layer in range(len(self.weights)-1,-1,-1):
            output = self.z[layer]
            weight = self.weights[layer]
            error = self.error_rates[layer+1]
            error = np.array(error)
            output = output.reshape(1,-1)
            print(error)
            print(output.shape)
            print(error.shape)
            print(weight.shape)
            print(self.find_derva_relu(output))
            self.error_rates[layer]=np.multiply(np.dot(error,weight.transpose()),self.find_derva_relu(output))
    def weight_bias_update(self):
        for layer in range(len(self.weights)-1,-1,-1):
            error = self.error_rates[layer+1]
            a = self.outputs[layer]
            weights = np.dot(a.reshape(-1,1),error)
            print('weight_update')
            print(a)
            print('outputs')
            print(error.shape)
            print(weights.shape)
            self.weight_error[layer] = np.add(self.weight_error[layer],weights)
            print(self.weight_error[layer])
        i = 0
        for bias in self.error_rates[1:len(self.error_rates)]:
            self.bias_error[i]=np.add(self.bias_error[i],bias.transpose())
            i+=1
        self.error_rates = []
        self.outputs = []
        self.z = []
            
    def predict(self,index):
        a = self.x_train[index]
        for layer in range(0,len(self.weights)):
            z = np.dot(a,self.weights[layer])
            z = np.add(self.biases[layer],z)
            print(z.shape)
            print('z')
            a = self.softmax(z)
        print('Predicted : '+str(a.argmax()))
        print('Actual : '+str(self.y_train[index]))
    def gradient_descent(self,t):
        for i in range(0,len(self.weights)):
            self.weights[i] = np.subtract(self.weights[i],(self.learning_rate/t)*self.weight_error[i])
        for i in range(0,len(self.biases)):
            self.biases[i] = np.subtract(self.biases[i],(self.learning_rate/t)*self.bias_error[i].flatten())
        self.weight_error = []
        self.bias_error = []
        for i in range(0,len(self.neurons)-1):
            self.weight_error.append(np.full((self.neurons[i],self.neurons[i+1]),0))
            self.bias_error.append(np.full((self.neurons[i+1],1),0))


            



            


            
        


        

akash = AkashNet()
akash.get_train_data()
akash.neuron_calc()
akash.random_weights_biases()
j = 1
t = 0
for i in range(0,500):
    l = 0
    for l in range(0,10):
        for k in range(int(j-1)*10,int(j*10)):
            t +=1
            akash.forward_prop(k)
            akash.find_error_rate(k)
            akash.backpropogate()  
            akash.weight_bias_update()
    akash.gradient_descent(t)
    t = 0
    j +=1  
for i in range(50001,50011):
    akash.predict(i)

