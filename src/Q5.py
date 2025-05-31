#!/usr/bin/env python3
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import pickle
import matplotlib.pyplot as plt
import cv2
import torch
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as functional
import sys, time
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor
from scipy.ndimage import filters
import rosbag
import roslib   #roslib.load_manifest(PKG)
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from scipy.ndimage import filters
from geometry_msgs.msg import Twist
PI = 3.1415926535897


training_file = "/home/max/assignment2/src/traindata.pkl"
validation_file = "/home/max/assignment2/src/validdata.pkl"
testing_file = "/home/max/assignment2/src/testdata.pkl"


with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

x_train, y_train = train['features'], train['labels']
x_valid, y_valid = valid['features'], valid['labels']
x_test, y_test = test['features'], test['labels']

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



classes = ('Stop', 'Right', 'Left', 'Forward', 'Roundabout')


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

#####################################################################################################
# Step 3: Defining loss function and optimizer
#-----------------------------------------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#####################################################################################################
# Step 4: Main training loop
#-----------------------------------------------

for epoch in range(6):    # we are using 5 epochs. Typically 100-200
  running_loss = 0.0
  for i, data in enumerate(train_loader, 0):
    # get the inputs
    inputs, labels = data

    # zero the parameter gradients
    optimizer.zero_grad()

    # Perform forward pass and predict labels
    predicted_labels = net(inputs)

    # Calculate loss
    loss = criterion(predicted_labels, labels)

    # Perform back propagation and compute gradients
    loss.backward()

    # Take a step and update the parameters of the network
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    if i % 300 == 299:    # print every 200 mini-batches
        print('Epoch: %d, %5d loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
        running_loss = 0.0

print('Finished Training.')
torch.save(net.state_dict(), "traffic_signs_classifier.pt")
torch.save(net, '/home/max/assignment2/cifar_2.pt')
print('Saved model parameters to disk.')

#####################################################################################################
# Step 5: Evaluate the accuracy of your network on the test dataset
#--------------------------------------------------------------------

model = Network()
model.load_state_dict(torch.load('traffic_signs_classifier.pt'))
model.eval()

class image_feature:

    def __init__(self):
        # subscribed Topic
        self.subscriber = rospy.Subscriber('/raspicam_node/image/compressed',
            CompressedImage,self.callback,queue_size=1)

    def callback(self, ros_data):
        velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=1)
        angular_speed = 90*2*PI/360
        relative_angle = 90*2*PI/360
        vel_msg = Twist()
        vel_msg.linear.x = 0.1
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0
        velocity_publisher.publish(vel_msg)

        font = cv2.FONT_HERSHEY_SIMPLEX
        #### direct conversion to CV2 ####
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  #original image
        image_np = cv2.rotate(image_np, cv2.ROTATE_180)    #rotate
        alpha = 0.7     # change the contrast
        beta = 0

        # eliminate glare
        R, G, B = cv2.split(image_np)
        output1_R = cv2.equalizeHist(R)
        output1_G = cv2.equalizeHist(G)
        output1_B = cv2.equalizeHist(B)
        equ = cv2.merge((output1_R, output1_G, output1_B))
        res = np.hstack((image_np,equ)) #stacking images side-by-side

        # change the contrast use to test the redline
        adjusted = cv2.convertScaleAbs(image_np, alpha=alpha, beta=beta)
        HSVImage = cv2.cvtColor(adjusted,cv2.COLOR_BGR2HSV) # for redline detection
        HSVImage_imp = cv2.cvtColor(res,cv2.COLOR_BGR2HSV)  #for traffic signal recognion

        Low_red_range = np.array([140,50,50])
        High_red_range = np.array([180,255,255])
        Low_white_range = np.array([80,35,70])
        High_white_range = np.array([135,230,255])

        mask_red = cv2.inRange(HSVImage,Low_red_range,High_red_range)
        mask_red = cv2.medianBlur(mask_red, 7)  # Median fifler
        contours2, hierarchy2 = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt2 in contours2:
            (x2, y2, w2, h2) = cv2.boundingRect(cnt2)
            if w2>3*h2 and (y2==307 or y2 == 308 or y2 == 312 or y2== 310): # take one flame each signal
                crop_img = HSVImage_imp[200:310,190:490]
                mask_white = cv2.inRange(crop_img,Low_white_range,High_white_range)
                mask_white = cv2.medianBlur(mask_white, 7)   # Median fifler
                mask_red1 = cv2.inRange(crop_img,Low_red_range,High_red_range)
                mask_red1 = cv2.medianBlur(mask_red1, 7)   # Median fifler
                mask = cv2.bitwise_or(mask_red1,mask_white)
                cv2.imshow('mask_white', mask)
                contours1, hierarchy1 = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for cnt1 in contours1:
                    (x, y, w, h) = cv2.boundingRect(cnt1)
                crop_img_1 = crop_img[y:y+60,x:x+120]
                BGR_img_1 = cv2.cvtColor(crop_img_1,cv2.COLOR_HSV2BGR)
                cv2.imwrite('traffic_image_original_1.png',BGR_img_1)
                resize_img = cv2.resize(BGR_img_1,(32,32))
                cv2.imwrite('traffic_image_2.png',resize_img)

                # use for model detect
                resize_img_1 = resize_img[:,:,[2,1,0]]
                x = ToTensor()(resize_img_1)
                x = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))(x)
                x.unsqueeze_(0)
                outputs = model(x)
                sm = nn.Softmax(dim=1)
                sm_outputs = sm(outputs)
                _, predicted = torch.max(sm_outputs,dim=1)
                print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                        for j in range(1)))
                sign = classes[predicted]
                cv2.putText(image_np,format(sign), (240,220),font,2, (0,0,0),2, cv2.LINE_AA)

                if sign == "Stop":
                    vel_msg.linear.x = 0
                    vel_msg.linear.y = 0
                    vel_msg.linear.z = 0
                    vel_msg.angular.x = 0
                    vel_msg.angular.y = 0
                    vel_msg.angular.z = 0
                    velocity_publisher.publish(vel_msg)

                elif sign == "Right":
                    current_angle = 0
                    t0 = rospy.Time.now().to_sec()
                    while(current_angle < relative_angle):
                        vel_msg.angular.z = -abs(angular_speed)
                        velocity_publisher.publish(vel_msg)
                        t1 = rospy.Time.now().to_sec()
                        current_angle = angular_speed*(t1-t0)

                elif sign == "Left":
                    current_angle = 0
                    t0 = rospy.Time.now().to_sec()
                    while(current_angle < relative_angle):
                        vel_msg.angular.z = abs(angular_speed)
                        velocity_publisher.publish(vel_msg)
                        t1 = rospy.Time.now().to_sec()
                        current_angle = angular_speed*(t1-t0)

                elif sign == "Roundabout":
                    current_angle = 0
                    t0 = rospy.Time.now().to_sec()
                    while(current_angle < relative_angle*2):
                        vel_msg.angular.z = -abs(angular_speed)
                        velocity_publisher.publish(vel_msg)
                        t1 = rospy.Time.now().to_sec()
                        current_angle = angular_speed*(t1-t0)
        cv2.imshow('cv_img', image_np)
        cv2.waitKey(1)


def main(args):
    ic = image_feature()
    rospy.init_node('image_feature', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
            print ("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
