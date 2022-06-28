import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

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


device = 'cuda' if torch.cuda.is_available() else 'cpu' #device selection

# random seed fix
torch.manual_seed(777)

# random seed fix for gpu
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

learning_rate = 0.001
training_epochs = 15
batch_size = 50


mnist_train = dsets.MNIST(root='MNIST_data/', # download path
                          train=True, # True=training data download
                          transform=transforms.ToTensor(), # transform to tensor
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/', 
                         train=False, # False=test data download
                         transform=transforms.ToTensor(), 
                         download=True)

#data loading
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True) #when the last batch is less than batch size, loader do not use last batch data

# CNN model call
model = CNN().to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)    # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #optimizer

total_batch = len(data_loader)
print('total batch size : {}'.format(total_batch))


for epoch in range(training_epochs):
    avg_cost = 0 #cost(loss) variable

    for X, Y in data_loader: # X=mini batch, Y=label
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad() #gradient intialize
        hypothesis = model(X) #prediction
        cost = criterion(hypothesis, Y) #loss calculation
        cost.backward() #backward propagation
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))




