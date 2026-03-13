import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import h5py

# Define your loss function here
class Lossfunc(object):

# This reads the matlab data from the .mat file provided
class MatRead(object):
    def __init__(self, file_path):
        super(MatRead).__init__()

        self.file_path = file_path
        self.data = h5py.File(self.file_path)

    def get_strain(self):
        strain = np.array(self.data['strain']).transpose(2,0,1)
        return torch.tensor(strain, dtype=torch.float32)

    def get_stress(self):
        stress = np.array(self.data['stress']).transpose(2,0,1)
        return torch.tensor(stress, dtype=torch.float32)

# Define data normalizer
class DataNormalizer(object):

# Define network your neural network for the constitutive model below
class Const_Net(nn.Module):

######################### Data processing #############################
# Read data from .mat file
path = 'data/Material_A.mat' #Define your data path here
data_reader = MatRead(path)
strain = data_reader.get_strain()
stress = data_reader.get_stress()

# Split data into train and test
ntrain =  # Specify the training data
ntest =   # Specify the test data
train_strain =
train_stress =
test_strain =
test_stress =

# Normalize your data
strain_normalizer   =
train_strain_encode =   # this should be the data after normalization
test_strain_encode  =

stress_normalizer   =
train_stress_encode =
test_stress_encode  =

ndim = strain.shape[2]  # Number of components
nstep = strain.shape[1] # Number of time steps
dt = 1/(nstep-1)

# Create data loader
batch_size = 20
train_set = Data.TensorDataset(train_strain, train_stress)
train_loader = Data.DataLoader(train_set, batch_size, shuffle=True)

############################# Define and train network #############################
# Create Nueral network, define loss function and optimizer
net = Const_Net()   # specify your neural network architecture

n_params = sum(p.numel() for p in net.parameters() if p.requires_grad) #Calculate the number of training parameters
print('Number of parameters: %d' % n_params)

loss_func = Lossfunc() #define loss function
optimizer =  # define optimizer
scheduler =  # define scheduler

# Train network
epochs =    # define number of training epochs
print("Start training for {} epochs...".format(epochs))

loss_train_list = []
loss_test_list = []

for epoch in range(epochs):
    net.train(True)
    trainloss = 0


    for i, data in enumerate(train_loader):
        input, target = data
        #define forward neural network evaluation below
        output_encode = # Forward
        output        = # Decode output
        loss = loss_func() # Calculate loss

                             # Clear gradients
                             # Backward
                             # Update parameters
                             # Update learning rate

        trainloss +=         # update your train loss here

    # Compute your test loss below
    net.eval()
    with torch.no_grad():

    # Print train loss every 10 epochs
    if epoch % 10 == 0:
        print("epoch:{}, train loss:{}, test loss:{}".format(epoch, trainloss/len(train_loader), testloss))

    # Save loss
    loss_train_list.append(trainloss/len(train_loader))
    loss_test_list.append(testloss)


print("Train loss:{}".format(trainloss/len(train_loader)))
print("Test loss:{}".format(testloss))

############################# Plot your result below using Matplotlib #############################
plt.figure(1)
plt.title('Train and Test Losses')

plt.figure(2)
plt.title('Truth Stresses vs Approximate Stresses for Sample {}')