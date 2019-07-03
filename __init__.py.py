
#We want to train a convolutional neural network to recognise
#handwritten digits from the MNIST database.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as tr
import matplotlib.pyplot as plt

#These are hyperparameters.
learning_rate = 0.01
loss_function = nn.MSELoss() #(Mean Squared Error)
batch_size_training = 100
batch_size_test = 100
epochs = 3

#This transform will convert the MNIST data to a tensor, then normalize it.
mnist_transform = tr.Compose([tr.ToTensor(),
                             tr.Normalize((0.1307,),(0.3081,))
                              ])

#Here we define the datasets, one for testing and one for training.
mnist_traindata = datasets.MNIST(root="./data", train=True, download=True,
                                transform=mnist_transform)

mnist_testdata = datasets.MNIST(root="./data", train=False, download = True,
                                transform=mnist_transform)

#DataLoaders are objects we can iterate over to obtain batches of data
#from the dataset we pass as the first argument as multivariate tensors.
train_loader = torch.utils.data.DataLoader(
    mnist_traindata, batch_size = batch_size_training, shuffle = True)


test_loader = torch.utils.data.DataLoader(
    mnist_testdata, batch_size = batch_size_test, shuffle = True)


"""
"One-Hot encoding" avoids innappropriate rewards to the network.
Consider the case where the network is presented with an image
of the digit "5". We wish to describe the difference between
the output of the network and 5, the "ground truth". The naive
approach is to compare the outputted prediction of the network
(from 0 to 9) to the ground-truth (0-9). But the network is
EQUALLY wrong about the identity of the glyph in the case where 
it predicts a 4 and the case where it predicts a 2. To equalise
the mean squared difference in these senarios, we can encode a 
5 as a 10-vector like so: [0,0,0,0,0,1,0,0,0,0], where the index
of the 1 corresponds to the encoded digit.
Luckily, tensors in pytorch are mutable, so this is trivial: 
"""
def as_tensor(digits):
        target_tensor = torch.zeros(batch_size_training,10)
        for i in range(len(digits)):
                target_tensor[i][digits[i]] = 1
        return target_tensor

def as_digit(tensor):
    #the Tensor.max method returns a tuple of the highest value and its index
    #In our case we only care about the index
    value, index = torch.max(tensor,0)
    return index


################################################################

class Net(nn.Module):
    def __init__(self):
        #First we inherit functionality from the nn.Module class
        #this has lots of useful things in it, like a __call__
        #method that automatically calls a Net.forward method
        super(Net, self).__init__()
        #Now we define some convolution layers:
        #for good intuition on how these work without the mathematical formalism,
        #check out https://www.youtube.com/watch?v=C_zFhWdM4ic
        #and then https://www.youtube.com/watch?reload=9&v=py5byOOHZM8

        #1 input channel (image), 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1,6,3)
        #6 input channels, 16 output channels, 3x3 square conv kernel
        self.conv2 = nn.Conv2d(6,16,3)
        #Some traditional linear layers now:
        #16 channels, each of which is 5x5 pixels, for 16*6*6 inputs
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120,84)
        #Our output layer is a 10-vector
        self.fc3 = nn.Linear(84,10)
        self.success = 0
        self.fail = 0
        self.accuracies = []
    def forward(self,x):
        """
        The basic function of the CNN.
        Takes in a tensor and outputs a predicted tensor
        :param x: the input tensor
        :return: a tensor as a prediction
        """
        #relu is "rectified linear unit function" ie ramp(x)
        #max_pool2d(M,2) takes each contiguous 2x2 square of elements
        #of the AxA matrix M and returns an (A/2)*(A/2) matrix
        #composed of the max element of each of those squares.
        #Used to prevent overfitting (prevents superfine detail
        #dominating the model by abstracting it out)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        #This is flattening x to a vector (from a 2d matrix)
        #The -1 is automatically converted to 1
        #producing a list-like structure...
        x = x.view(-1,self.num_flat_features(x))
        #...which is then thrown into a pair of linear layers
        #with a ramp(x) - "ReLU" - activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #And finally the result is just the last layer
        #(which is a 10-vector)
        x = self.fc3(x)
        #x contains metadata, tracking all the operations that
        #have been performed on it. This is used to calculate the
        #gradient, to perform gradient descent on the loss function.
        return x
    def num_flat_features(self,x):
        """
        this is a helper function for line 106
        :param x: multidimensional torch.Tensor
        :return: the length of x if flattened to a vector
        """
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features*=s
        return num_features
    def train_cycle(self,input,target):
        """
        This is a single training cycle for the network.
        It takes in one batch of inputs and compares
        them to one batch of targets ("ground-truths")
        :param input: training data as a torch.Tensor
        :param target: goals as a torch.Tensor
        :return: output of network as torch.Tensor
        """
        output = self(input)
        target_tensors = as_tensor(target)
        loss = loss_function(output, target_tensors)
        net.zero_grad()
        self.success, self.fail = 0, 0
        loss.backward()
        for i in range(batch_size_training):
                if as_digit(output[i]) == target[i].item():
                    self.success += 1
                else:
                    self.fail += 1
        for f in self.parameters():
            f.data.sub_(f.grad.data*learning_rate)
    def train_epoch(self,loader):
        for batch_idx, (data, targets) in enumerate(train_loader):
            self.train_cycle(data, targets)
            if batch_idx%20 == 0:
                net.log()
    def log(self):
        try:
            percent = (100*self.success/(self.fail+self.success))
        except ZeroDivisionError:
            percent = 0
        print("Accuracy: %.2f%%" %(percent))
        self.accuracies.append(percent)





net = Net()
for i in range(epochs):
        net.train_epoch(train_loader)
        
#After training, we can plot the accuracy of the network
plt.plot(net.accuracies)
plt.xlabel("Number of Batches seen")
plt.ylabel("Percent accuracy")
plt.title("CNN performance on MNIST data")
plt.show()





