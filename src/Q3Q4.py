#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
import matplotlib.pyplot as plt
import cv2
import torch
import time
import PIL
from PIL import Image
import numpy as np
import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as functional

from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import functional as fn


import rosbag
import roslib   #roslib.load_manifest(PKG)
import rospy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from scipy.ndimage import filters

font = cv2.FONT_HERSHEY_SIMPLEX

training_file = "/home/ob1-knb/ur5/src/traindata.pkl"
validation_file = "/home/ob1-knb/ur5/src/validdata.pkl"
testing_file = "/home/ob1-knb/ur5/src/testdata.pkl"
xtra_file = "/home/ob1-knb/ur5/src/extradata.pkl"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
with open(xtra_file, mode='rb') as f:
    xtra = pickle.load(f)


x_train, y_train = train['features'], train['labels']
x_valid, y_valid = valid['features'], valid['labels']
x_test, y_test = test['features'], test['labels']
x_xtra, y_xtra = xtra['features'], xtra['labels']

# Custom Dataloader Class
class TrafficSignsDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.features[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)
            y = y

        return x, y


# Define custom transforms to apply to the images
# ------------------------------------------------
# Here we are applying a normalization to bring all pixel values between 0 and 1
# and converting the data to a pytorch tensor
custom_transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.5,0.5,0.5 ), (0.5,0.5,0.5))])

# Define the dataloaders
# -----------------------
# Dataloaders help in managing large amounts of data efficiently
batch_size = 5

train_dataset = TrafficSignsDataset(x_train, y_train,
                                    custom_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

validation_dataset = TrafficSignsDataset(x_valid, y_valid,
                                         custom_transform)
validation_loader = DataLoader(validation_dataset, shuffle=True)

test_dataset = TrafficSignsDataset(x_test, y_test,
                                   custom_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

xtra_dataset = TrafficSignsDataset(x_xtra, y_xtra,
                                   custom_transform)
xtra_loader = DataLoader(xtra_dataset, batch_size=batch_size, shuffle=True)



classes = ('Stop', 'Right', 'Left', 'Forward', 'Roundabout')

def imshow(img):
    img = img / 2 + 0.5

dataiter = iter(train_loader)
images, labels = dataiter.next()

print("Number of training samples is {}".format(len(train_dataset)))
print("Number of test samples is {}".format(len(test_dataset)))
print("Number of validation samples is {}".format(len(validation_dataset)))

print("Batch size is {}".format(len(images)))
print("Size of each image is {}".format(images[0].shape))

print("The labels in this batch are: {}".format(labels))
print("These correspond to the classes: {}, {}, {}, {}, {}".format(
    classes[labels[0]], classes[labels[1]],
    classes[labels[2]], classes[labels[3]], classes[labels[4]]))

#####################################################################################################
# Step 2: Building your Neural Network
#---------------------------------------

class Network(nn.Module):
  def __init__(self):
    self.output_size = 5   # 5 classes

    super(Network, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)             # 2D convolution
    self.pool = nn.MaxPool2d(2, 2)              # max pooling
    self.conv2 = nn.Conv2d(6, 16, 5)            # 2D convolution
    self.fc1 = nn.Linear(16 * 5 * 5, 120)       # Fully connected layer
    self.fc2 = nn.Linear(120, 84)               # Fully connected layer
    self.fc3 = nn.Linear(84, self.output_size)  # Fully connected layer

  def forward(self, x):
    """Define the forward pass."""
    x = self.pool(functional.relu(self.conv1(x)))
    x = self.pool(functional.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = functional.relu(self.fc1(x))
    x = functional.relu(self.fc2(x))
    x = self.fc3(x)
    return x

net = Network()

#####################################################################################################
# Step 3: Defining loss function and optimizer
#-----------------------------------------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#####################################################################################################
# Step 4: Main training loop
#-----------------------------------------------

for epoch in range(5):
  running_loss = 0.0
  val_loss = 0.0

  for i, data in enumerate(validation_loader, 0):

    inputs, labels = data
    optimizer.zero_grad()
    predicted_labels = net(inputs)
    loss = criterion(predicted_labels, labels)
    loss.backward()
    optimizer.step()

    val_loss += loss.item()
    if i % 200 == 199:    # print every 200 mini-batches
        print('Epoch: %d, %5d val loss: %.3f' % (epoch + 1, i + 1, val_loss / 50))
        val_loss = 0.0

  for i, data in enumerate(train_loader, 0):

    inputs, labels = data
    optimizer.zero_grad()
    predicted_labels = net(inputs)
    loss = criterion(predicted_labels, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if i % 200 == 199:
        print('Epoch: %d, %5d loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
        running_loss = 0.0

print('Finished Training.')
torch.save(net.state_dict(), "traffic_signs_classifier.pt")
print('Saved model parameters to disk.')

#####################################################################################################
# Step 5: Duckietown Experiment
#--------------------------------------------------------------------

model = Network()
model.load_state_dict(torch.load('traffic_signs_classifier.pt'))
model.eval()

bag_file = '/home/ob1-knb/ur5/src/bag1.bag'
bag = rosbag.Bag(bag_file, "r")

bridge = CvBridge()
bag_data = bag.read_messages('/raspicam_node/image/compressed')

for topic, msg, t in bag_data:
    cv_image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

    dst = cv2.rotate(cv_image, cv2.ROTATE_180)
    original = dst
    alpha = 0.5
    beta = 0
    res = cv2.convertScaleAbs(dst,alpha=alpha,beta=beta)
    HSVImage_imp = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
    Low_red_range = np.array([140,50,50])
    High_red_range = np.array([180,255,255])
    Low_white_range = np.array([100,43,46])
    High_white_range = np.array([124,255,255])

    mask_red = cv2.inRange(HSVImage_imp,Low_red_range,High_red_range)
    mask_red = cv2.medianBlur(mask_red, 7)

    mask_red, contours2, hierarchy2 = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt2 in contours2:
        (x2, y2, w2, h2) = cv2.boundingRect(cnt2)

        if  (y2>308 and y2<315) and w2 > 3*h2 :

            crop_img = HSVImage_imp[220:300,130:490]
            mask_white = cv2.inRange(crop_img,Low_white_range,High_white_range)
            mask_white = cv2.medianBlur(mask_white, 7)  # 中值滤波
            mask_red1 = cv2.inRange(crop_img,Low_red_range,High_red_range)
            mask_red1 = cv2.medianBlur(mask_red1, 7)  # 中值滤波
            mask = cv2.bitwise_or(mask_red1,mask_white)
            mask_white, contours1, hierarchy1 = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for cnt1 in contours1:
                (x, y, w, h) = cv2.boundingRect(cnt1)
                crop_img_1 = crop_img[y:y+60,x:x+120]
                RGB_img_1 = cv2.cvtColor(crop_img_1,cv2.COLOR_HSV2BGR)

            resize_img = cv2.resize(RGB_img_1,(32,32))


            resize_img = resize_img[:, :, [2,1,0]]
            # resize_img = resize_img[:,:,::-1]

            RGB_img_1 = cv2.resize(RGB_img_1,(100,100))

            cv2.imshow("Image window",RGB_img_1)

            x = ToTensor()(resize_img)
            x = transforms.Normalize((0.5,0.5,0.5 ), (0.5,0.5,0.5))(x)
            x.unsqueeze_(0)

            outputs = model(x)
            sm = nn.Softmax(dim=1)
            sm_outputs = sm(outputs)

            probs, index = torch.max(sm_outputs, dim=1)
            for p, i in zip(probs, index):
                print('{}'.format(classes[i]))
                sign = classes[i]
                cv2.putText(original, '{}'.format(sign), (240,220), font, 2, (0,0,0), 2, cv2.LINE_AA)
                time.sleep(0.5)

    cv2.imshow("Image window", original)


    cv2.waitKey(3)

#####################################################################################################
# Step 6: Evaluate the accuracy of your network on the test dataset
#--------------------------------------------------------------------

dataiter = iter(train_loader)
images, labels = dataiter.next()

fig, ax = plt.subplots(1, len(images))
for id, image in enumerate(images):
  # convert tensor back to numpy array for visualization
  ax[id].imshow((image / 2 + 0.5).numpy().transpose(1,2,0))
  ax[id].set_title(classes[labels[id]])
plt.show()
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

outputs = model(images)

sm = nn.Softmax(dim=1)
sm_outputs = sm(outputs)

probs, index = torch.max(sm_outputs, dim=1)
for p, i in zip(probs, index):
    print('True label {0}, Predicted label {0} - {1:.4f}'.format(classes[i], p))

correct = 0
total = 0
with torch.no_grad():
    for data in validation_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Correct predictions: {}'.format(correct))
print('Total predicted: {}'.format(total))
print('Number per class: ')
print(stop)
print(right)
print(left)
print(forward)
print(round)
print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
