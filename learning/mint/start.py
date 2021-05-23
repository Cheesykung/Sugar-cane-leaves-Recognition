import pandas as pd
import numpy as np
from Net import Net
# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt
# for creating validation set
from sklearn.model_selection import train_tes
t_split
# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm
# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntro
pyLoss, Sequential, Conv2d, MaxPool2d, Module
, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

train = pd.read_csv
(
‘C:/train/train.csv')
train.head()
train_img = []
for img_name in tqdm(train['id']):
# defining the image path
impath = 'C:/train/'+str(img_name)+'.png
’
# reading the image
img = imread
(impath, as_gray=True)
# normalizing the pixel values
img /= 255.0
# converting the type of pixel to float
img = img.astype('float32
’
)
# appending the image into the list
train_img.append
(img
)
# converting the list to numpy array
train_x = np.array
(train_img
)
# defining the target
train_y = train['label'].values
print(train_x.shape
)
# print >>> (60000, 28, 28
)
train_x, val_x, train_y, val_y = 
train_test_split(train_x, train_y, test_size = 0.1)
# converting training images into torch format
train_x = train_x.reshape(54000, 1, 28, 28)
train_x = torch.from_numpy(train_x).to(torch.float32)
# converting the target into torch format
train_y = train_y.astype(int);
train_y = torch.from_numpy(train_y).to(torch.float32)
# shape of training data
print(train_x.shape, train_y.shape)
# converting validation images into torch format
val_x = val_x.reshape(6000, 1, 28, 28)
val_x = torch.from_numpy(val_x).to(torch.float32)
# converting the target into torch format
val_y = val_y.astype(int);
val_y = torch.from_numpy(val_y).to(torch.float32)
# shape of validation data
print(val_x.shape, val_y.shape)

# defining the model
model = Net()
# defining the optimizer
# optimizer = Adam(model.parameters(), lr=0.07)
optimizer = SGD(model.parameters(), lr=0.07, momentum=0.9)
# defining the loss function
criterion = CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
model = model.cuda()
criterion = criterion.cuda()
print(model)

# empty list to store training losses
train_losses = []
# empty list to store validation losses
val_losses = []
# defining the number of epochs
n_epochs = 25
# training the model
for epoch in tqdm(range(n_epochs)):
model.train()
tr_loss = 0
# getting the training set
x_train, y_train = Variable(train_x), Variable(train_y)
# getting the validation set
x_val, y_val = Variable(val_x), Variable(val_y)
# clearing the Gradients of the model parameters
optimizer.zero_grad()
# prediction for training and validation set
output_train = model(x_train)
output_val = model(x_val)

# computing the training and validation loss
# we convert the results because they aren't in 
the good format
y_train = y_train.long() 
y_train = y_train.squeeze_()
y_val = y_val.long() 
y_val = y_val.squeeze_()
loss_train = criterion(output_train, y_train)
loss_val = criterion(output_val, y_val)
train_losses.append(loss_train)
val_losses.append(loss_val)
# computing the updated weights of all the model parameters
loss_train.backward()
optimizer.step()
tr_loss = loss_train.item()

# computing the training and validation loss
# we convert the results because they aren't in 
the good format
y_train = y_train.long() 
y_train = y_train.squeeze_()
y_val = y_val.long() 
y_val = y_val.squeeze_()
loss_train = criterion(output_train, y_train
)
loss_val = criterion(output_val, y_val
)
train_losses.append
(loss_train
)
val_losses.append
(loss_val
)
# computing the updated weights of all the model parameters
loss_train.backward()
optimizer.step()
tr_loss = loss_train.item()

# Saving Entire Model 
# A common PyTorch convention is to save models 
using either a .pt or .pth file extension.
torch.save(model, 'C:/model04.pt')

# prediction for validation set
# don’t need gradients in test step since the 
parameter updates has been done in training 
step. Using ‘torch.no_grad()’ in the test and 
validation phase yields the faster 
inference(speed up computation) and reduced 
memory usage(which allows us to use larger 
size of batch).
with torch.no_grad():
#output = model(val_x.cuda())
output = model(val_x)
softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)
# accuracy on validation set
accuracy_score(val_y, predictions

import pandas as pd
import numpy as np
from Net import Net
# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt
# for creating validation set
from sklearn.model_selection import train_test_split
# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm
# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss
, Sequential, Conv2d, MaxPool2d, Module, Softmax, B
atchNorm2d, Dropout
from torch.optim import Adam, SGD