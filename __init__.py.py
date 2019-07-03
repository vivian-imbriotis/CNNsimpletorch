#A convolution neural network, with many notes for you
#my friend Future Viv.

#We want to train a convolutional neural network to recognise
#handwritten digits from the MNIST database

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as tr
import matplotlib.pyplot as plt


learning_rate = 0.01
loss_function = nn.MSELoss() #(Mean Squared Error)

#Because I'm a big dumb binch, we're using stochastic gradient descent
#rather than batch gradient descent. When I'm cleverer I'll figure that
#one out

#Lol jk figuring out batch gradient descent now

batch_size_training = 100
batch_size_test = 100
epochs = 3


mnist_transform = tr.Compose([tr.ToTensor(),
                             tr.Normalize((0.1307,),(0.3081,))
                              ])


mnist_traindata = datasets.MNIST(root="./data", train=True, download=True,
                                transform=mnist_transform)

mnist_testdata = datasets.MNIST(root="./data", train=False, download = True,
                                transform=mnist_transform)


train_loader = torch.utils.data.DataLoader(
    mnist_traindata, batch_size = batch_size_training, shuffle = True)


test_loader = torch.utils.data.DataLoader(
    mnist_testdata, batch_size = batch_size_test, shuffle = True)






#28x28 pixel images





def as_tensor(digits):
        target_tensor = torch.zeros(batch_size_training,10)
        for i in range(len(digits)):
                target_tensor[i][digits[i]] = 1
        return target_tensor


def as_digit(tensor):
    value, index = torch.max(tensor,0)
    return index


################################################################

class Net(nn.Module):
    def __init__(self):
        #First we inherit functionality from the nn.Module class
        super(Net, self).__init__()
        #Now we define some convolution layers:
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
        #with a ramp(x) activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #And finally the result is just the last layer
        #(which is a 10-vector)
        x = self.fc3(x)
        #x contains metadata, tracking all the operations that
        #have been performed on it. This means we can call a single
        #function on x to do calculate the gradient
        return x
    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features*=s
        return num_features
    def train_cycle(self,input,target):
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
        

plt.plot(net.accuracies)
plt.show()





