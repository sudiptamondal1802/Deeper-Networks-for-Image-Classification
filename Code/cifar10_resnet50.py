""" Setting up environment """
import torch
import torchvision
from torchvision import datasets
import os
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import time
import datetime
import sys
import pickle
import pandas as pd
import seaborn as sns

""" Import libraries for building network """
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet50
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)
# os.chdir("/content/drive/My Drive")

#DATA DIRECTORY PATH
DATA_PATH = 'Data'
#SAVE PATH FOR MODEL,IMAGES
SAVE_PATH = 'CIFAR10_RESNET50/'
#PICKLE DUMP FOLDER FOR TRAIN & VALID LOSS & ACCURACY
PICKLE_DUMP = 'PICKLE_DUMP/'

#if the path does not exist create 
if not os.path.exists(PICKLE_DUMP):
    os.mkdir(PICKLE_DUMP)
#if the path does not exist create 
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

    
    
""" CHECK AND ASSIGN CUDA """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')


"""Parameters"""
n_epochs = 50
learning_rate = 0.01
momentum = 0.9
log_interval = 10

""" Load dataset"""
# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 128
train_data = torchvision.datasets.CIFAR10(DATA_PATH, train=True, download=True,
                                        transform=torchvision.transforms.Compose([
                                         torchvision.transforms.RandomCrop(32, padding=4),
                                         torchvision.transforms.RandomHorizontalFlip(),
                                         transforms.RandomRotation(10), 
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]))

test_data = torchvision.datasets.CIFAR10(DATA_PATH, train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                         torchvision.transforms.RandomCrop(32, padding=4),
                                         torchvision.transforms.RandomHorizontalFlip(),
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]))

train_size = int(0.8 * len(train_data))   #80% training data
valid_size = len(train_data) - train_size #20% validation data
train_dataset, valid_dataset = torch.utils.data.random_split(train_data, [train_size, valid_size])

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size,
                                           num_workers=num_workers)
valid_loader   = torch.utils.data.DataLoader(dataset=valid_dataset, shuffle=False, batch_size=batch_size,
                                           num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=20,
                                          num_workers=num_workers)


print("Length of training dataset:", len(train_dataset))
print("Length of validation dataset:", len(valid_dataset))
print("Length of test dataset:", len(test_data))

print("Length of the train_loader:", len(train_loader.dataset))
print("Length of the valid_loader:", len(valid_loader.dataset))
print("Length of the test_loader:", len(test_loader.dataset))

""" Visualise some of the samples"""
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

# # get the shape of the example data
# example_data.shape

# specify the image classes for CIFAR10
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image


# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()  # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure()
# display 10 images
for idx in np.arange(10):
    ax = fig.add_subplot(2, 10 / 2, idx + 1, xticks=[], yticks=[])
    plt.tight_layout()
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])
# save the images
plt.savefig(SAVE_PATH + 'Cifar10_Resnet50_images.png')
plt.close()

""" Build Network"""
""" Resnet50"""
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        # define model
        self.model = resnet50(pretrained=False, num_classes=10)
        
    def forward(self, x):
        return self.model(x)


""" Initialize network,scheduler, loss and optimizer"""
network = ResNet50()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 5)
loss_function = nn.CrossEntropyLoss()

# move tensors to GPU if CUDA is available
if train_on_gpu:
    network.to(device)

""" Training the model"""

#INITIALIZE VARIABLES
train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []
best_valid_acc = 0.0
train_hist = {}
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

#START THE TRAINING AND EVALUATION
print('Logging started......check log file and images in the folder {} after end of execution'.format(SAVE_PATH))
if os.path.exists(os.path.join(SAVE_PATH,"logfile.txt")):
    os.remove(os.path.join(SAVE_PATH,"logfile.txt"))
    
class print_to_file(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() 
    def flush(self) :
        for f in self.files:
            f.flush()

logfile = open(os.path.join(SAVE_PATH,"logfile.txt"), 'a')
original = sys.stdout
sys.stdout = print_to_file(sys.stdout, logfile)

# Training and evaluation on validation set record the start time
start_time = time.time()
now = datetime.datetime.now()
print("Start time of the training:",now.strftime("%Y-%m-%d %H:%M:%S"))

for epoch in range(1, n_epochs + 1):
    # start time of the epoch
    epoch_start_time = time.time()

    # intialize the train and validation loss and accuracy for the epoch
    train_loss = 0.0
    train_accuracy = 0.0
    valid_loss = 0.0
    valid_accuracy = 0.0

    # train the model
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        train_accuracy += pred.eq(target.data.view_as(pred)).sum()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

    #Average train loss and total accuracy for the epoch
    train_loss /=len(train_loader.dataset)  #average training loss
    train_accuracy_epoch = 100. * train_accuracy.cpu()/len(train_loader.dataset)
    
    # end of the training
    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    # evaluate on validation data
    network.eval()
    with torch.no_grad():
        for  batch_idx, (data, target) in enumerate(valid_loader):
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.to(device), target.to(device)
            output = network(data)
            valid_loss += loss_function(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            valid_accuracy += pred.eq(target.data.view_as(pred)).sum()

    #Average validation loss and total accuracy for the epoch
    valid_loss /=len(valid_loader.dataset) 
    valid_accuracy_epoch = 100. * valid_accuracy.cpu()/len(valid_loader.dataset)
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)  
    train_accuracies.append(train_accuracy_epoch)
    valid_accuracies.append(valid_accuracy_epoch) 

    curr_lr = optimizer.param_groups[0]['lr']
    
    # print training/validation statistics
    print('Epoch:',epoch)
    print('\nTraining set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    train_loss, train_accuracy, len(train_loader.dataset),train_accuracy_epoch))    
    print('\nValidation set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    valid_loss, valid_accuracy, len(valid_loader.dataset),valid_accuracy_epoch))
    print('\nLearning rate',curr_lr)
    
    scheduler.step(valid_loss/len(valid_loader))
    
    # if the best validation performance so far, save the network to file 
    if(best_valid_acc < valid_accuracy):
        best_valid_acc = valid_accuracy
        print('Saving best model')
        torch.save(network.state_dict(), SAVE_PATH + 'model.pth')
        torch.save(optimizer.state_dict(), SAVE_PATH + 'optimizer.pth')


end_time = time.time()
now = datetime.datetime.now()
print("End time of the training:",now.strftime("%Y-%m-%d %H:%M:%S"))
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)
print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (
np.mean(train_hist['per_epoch_ptimes']), n_epochs, total_ptime))

""" Evaluating the Model's Performance - Train and validation loss plot"""
plt.style.use('ggplot')
plt.plot(train_losses, label='Train loss')
plt.plot(valid_losses, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(SAVE_PATH + 'Cifar10_Resnet50_loss_plot.png')
plt.close()

""" Evaluating the Model's Performance - Train and validation accuracy plot"""
plt.style.use('ggplot')
plt.plot(train_accuracies, label='Train accuracy')
plt.plot(valid_accuracies, label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(SAVE_PATH + 'Cifar10_Resnet50_accuracy_plot.png')
plt.close()

"""# Test the trained network"""
print("Test results for Cifar10 Resnet50")
# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
nb_classes = 10
confusion_matrix = np.zeros((nb_classes, nb_classes))

network.eval()
# iterate over test data
for data, target in test_loader:
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.to(device), target.to(device)
    # forward pass: compute predicted outputs by passing inputs to the model
    output = network(data)
    # calculate the batch loss
    loss = loss_function(output, target)
    # update test loss
    test_loss += loss.item() * data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(20):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1
        
    for t, p in zip(target.view(-1), pred.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

plt.figure(figsize=(15,10))
df_cm = pd.DataFrame(confusion_matrix, index=classes, columns=classes).astype(int)
heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(SAVE_PATH + 'Cifar10_Resnet50_model2_cm.png')
plt.show()
plt.close()

# average test loss
test_loss = test_loss / len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

"""Visualise the model's output"""
# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()
images.numpy()

# move model inputs to cuda, if GPU available
if train_on_gpu:
    images = images.to(device)

# get sample outputs
output = network(images)
# convert output probabilities to predicted class
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20 / 2, idx + 1, xticks=[], yticks=[])
    imshow(images.cpu()[idx])
    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                 color=("green" if preds[idx] == labels[idx].item() else "red"))

plt.savefig(SAVE_PATH + 'Cifar10_Resnet50_sample_test_results.png')
plt.close()

#End the logging
sys.stdout = original
print('Logging ended....')
logfile.close()

"""Save the train and test losses for the plotting in pickle file"""
cifar10_resnet50_model2 = {
    "train_losses" :train_losses,  
    "valid_losses":valid_losses,
    "train_accuracies":train_accuracies,
    "valid_accuracies":valid_accuracies
    }

my_file = open(os.path.join(PICKLE_DUMP,'cifar10_resnet50_model2'), 'wb') 
my_file = pickle.dump(cifar10_resnet50_model2, my_file)
