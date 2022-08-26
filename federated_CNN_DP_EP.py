import os
import random
from tqdm import tqdm
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.datasets as dsets
torch.backends.cudnn.benchmark=True
import math
import matplotlib.pyplot as plt

def gaussian_noise(data_shape, clip_constant, epsilon, device=None):
    """
    Gaussian noise
    """
    sigma=math.sqrt(2*math.log(1.25/delta)/epsilon)
    return torch.normal(0, sigma * clip_constant, data_shape).to(device)


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First layer
        # ImgIn shape=(?, 28, 28, 1)(batch size, width, height, channel)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(), #activation function
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 두번째층
        # ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # fully connected layer 7x7x64 inputs -> 10 outputs
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)

        # weight initialization
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)   #Flatten, -1 means undetermined size(0) means first index
        out = self.fc(out)
        return out


def client_update(client_model, optimizer, train_loader, epoch, epsilon):
    """
    This function updates/trains client model on client data
    """
    model.train() # model for training
    criterion = torch.nn.CrossEntropyLoss().to(device)  
    for e in range(epoch):
    	clipped_grads = {name: torch.zeros_like(param) for name, param in client_model.named_parameters()} #noised_gradient variable initialization
    	for batch_idx, (data, target) in enumerate(train_loader):
    		data, target = data.to(device) , target.to(device) 
    		optimizer.zero_grad()
    		output = client_model(data)
    		loss = criterion(output, target)
    		loss.backward(retain_graph=True)
    		torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=clip_constant) #gradient clipping
    		
    		for name, param in client_model.named_parameters(): #allocation of current gradient to noised_gradient variable
    			clipped_grads[name] += param.grad 
    		#client_model.zero_grad()
    		
    		
    	for name, param in client_model.named_parameters():	#current gradient+gaussain_noise
    		clipped_grads[name] += gaussian_noise(clipped_grads[name].shape, clip_constant, epsilon, device)
    	for name, param in client_model.named_parameters(): 
    		clipped_grads[name]/=batch_size  
    	for name, param in model.named_parameters(): #allocation of noised gradient to model gradient
    		param.grad = clipped_grads[name]
    	optimizer.step()

            #print(loss.size()[0])
            #for name, param in model.named_parameters():
            #	print("name:",name,"param:",param.grad)

    return loss.item()


def server_aggregate(global_model, client_models):
    """
    This function has aggregation method 'mean'
    """
    global_dict = global_model.state_dict() #current model parameter 
    for k in global_dict.keys(): #iterate for whole model parameter
    #stack the entire client model parameter and average the parameter value for global model
    	global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    #	global_dict[k] += gaussian_noise(global_dict[k].shape, clip_constant, sigma, device) #adding gaussian_noise
    
    #update the global model parameter
    global_model.load_state_dict(global_dict)
    
    #update the client model parameter
    for model in client_models:
    	model.load_state_dict(global_model.state_dict())


def test(global_model, test_loader):
    """This function test the global model on test data and returns test loss and test accuracy """
    model.eval() # model for testing
    test_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss().to(device)  
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device) , target.to(device) 
            output = global_model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    return test_loss, acc



##### Hyperparameters for federated learning #########
num_clients = 5
num_selected = 5
epochs = 2
batch_size = 50
clip_constant=8
epsilon=[0.01,0.03,0.05,0.07,0.09]
delta=1E-7
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# random seed fix
torch.manual_seed(777)

# random seed fix for gpu
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

#learning_rate = 0.001	
training_epochs = 15


traindata = dsets.MNIST(root='MNIST_data/', # download path
                          train=True, # True=training data download
                          transform=transforms.ToTensor(), # transform to tensor
                          download=True)
testdata = dsets.MNIST(root='MNIST_data/', 
                         train=False, # False=test data download
                         transform=transforms.ToTensor(), 
                         download=True)


# Dividing the training data into num_clients, with each client having equal number of images
traindata_split = torch.utils.data.random_split(traindata, [int(traindata.data.shape[0] / num_clients) for _ in range(num_clients)])

#loading the data for training 
train_loader = [torch.utils.data.DataLoader(dataset=x, batch_size=batch_size, shuffle=True) for x in traindata_split]

#loading the data for testing
test_loader = torch.utils.data.DataLoader(dataset=testdata,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True) #when the last batch is less than batch size, loader do not use last batch data


global_model =  CNN().to(device)
client_models = [ CNN().to(device) for _ in range(num_clients)]

losses_train = []
losses_test = []
acc_train = []
acc_test = []


for model in client_models:
    model.load_state_dict(global_model.state_dict()) ### initial synchronizing with global model 

#optimizer
opt = [optim.Adam(model.parameters(), lr=0.001) for model in client_models]


for epoch in range(training_epochs):
    loss = 0
    client_idx = np.random.permutation(num_clients)[:num_selected] #random permutation for client selection
    for i in range(num_selected):
        loss += client_update(client_models[i], opt[i], train_loader[i], epochs, epsilon[i])
    losses_train.append(loss)

    server_aggregate(global_model, client_models)
    #test_loss, acc = test(client_models[0], test_loader)
    #test_loss1, acc1 = test(client_models[1], test_loader)
    #test_loss1, acc2 = test(client_models[2], test_loader)
    #test_loss1, acc3 = test(client_models[3], test_loader)
    #test_loss1, acc4 = test(client_models[4], test_loader)

    losses_test.append(test_loss)
    acc_test.append(acc)
    #print('%d-th round' % epoch)
    print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss / num_selected, test_loss, acc))
    #print(acc,acc1,acc2,acc3,acc4)

plt.title('Accurcy',fontsize=20)
plt.ylabel('Accuracy',fontsize=14)
plt.xlabel('Number of Epoch',fontsize=14)
plt.xticks(range(training_epochs))
plt.plot(range(training_epochs), acc_test, label='Accuracy', color='darkred')
plt.show()

dataiter = iter(test_loader)
images, labels = dataiter.next()
outputs = model(images)
_, predicted = torch.max(outputs, 1)
print('Predicted:', predicted, 'Labels:',labels)