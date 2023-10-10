import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, labels, deconvs, inputs, lamda=1):
        loss = nn.CrossEntropyLoss()(output, labels)
        loss += lamda * nn.MSELoss()(inputs, deconvs)
        return loss

class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.deconv1 = nn.ConvTranspose2d(16, 6, 5)
        self.deconv2 = nn.ConvTranspose2d(6, 3, 5)

    def forward(self, x):
        x, indices1 = self.pool(F.relu(self.conv1(x)))
        x, indices2 = self.pool(F.relu(self.conv2(x)))

        deconv = self.unpool(x, indices2)
        deconv = F.relu(deconv)
        deconv = self.deconv1(deconv)

        deconv = self.unpool(deconv, indices1)
        deconv = F.relu(deconv)
        deconv = self.deconv2(deconv)

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x, deconv


class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:
net1 = Net1()
net2 = Net2()
criterion1 = nn.CrossEntropyLoss()
criterion2 = CustomLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=0.001, momentum=0.9)
optimizer2 = optim.SGD(net2.parameters(), lr=0.001, momentum=0.9)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model_run = False

# functions to show an image


def imshow_task23(img, axs):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    axs.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()

def imshow_task1(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def task1():
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    imshow_task1(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
    PATH = './cifar_net.pth'
    if model_run:
        for epoch in range(2):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer1.zero_grad()

                # forward + backward + optimize
                outputs = net1(inputs)
                loss = criterion1(outputs, labels)
                loss.backward()
                optimizer1.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        print('Finished Training')
        torch.save(net1.state_dict(), PATH)
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    net1.load_state_dict(torch.load(PATH))
    outputs = net1(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                  for j in range(4)))

    # print images
    imshow_task1(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net1(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net1(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


def task2(lamda=10):
    PATH = './cifar_net2.pth'
    if model_run:
        for epoch in range(2):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer2.zero_grad()

                # forward + backward + optimize
                outputs, deconvs = net2(inputs)
                loss = criterion2(outputs, labels, deconvs, inputs)
                loss.backward()
                optimizer2.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        print('Finished Training')
        torch.save(net2.state_dict(), PATH)
    correct = 0
    total = 0
    net2.load_state_dict(torch.load(PATH))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs, deconvs = net2(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    outputs, deconvs = net2(images)
    _, predicted = torch.max(outputs, 1)
    fig, ax = plt.subplots(2, 1)
    fig.suptitle('Original and Reconstructed Images')
    ax[0].set_ylabel('Original_images')
    ax[1].set_ylabel('Reconstructed_images')
    imshow_task23(torchvision.utils.make_grid(images), ax[0])
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
    # show two-three examples of reconstructed images alongside the original images.
    imshow_task23(torchvision.utils.make_grid(deconvs), ax[1])
    plt.show()
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))


def task3():
    PATH = './cifar_net2.pth'
    trainloader3 = torch.utils.data.DataLoader(trainset, batch_size=1, num_workers=2, shuffle=True)
    testloader3 = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=2, shuffle=False)
    dataiter_train = iter(trainloader3)
    train_image, train_label = next(dataiter_train)
    dataiter_test = iter(testloader3)
    test_image, test_label = next(dataiter_test)
    net2.load_state_dict(torch.load(PATH))
    # create a torch grid of images, each for a different channel of the first convolutional layer.
    conv1_train, indices1_train = net2.pool(F.relu(net2.conv1(train_image)))
    conv1_test, indices1_test = net2.pool(F.relu(net2.conv1(test_image)))
    fig1, axs1 = plt.subplots(7, 1, figsize=(10, 10))
    fig2, axs2 = plt.subplots(7, 1, figsize=(10, 10))
    fig1.suptitle('Train_image - first convolutional layer')
    fig2.suptitle('Test_image - first convolutional layer')
    axs1[0].set_ylabel('original image', rotation=90, loc='center', labelpad=10)
    axs2[0].set_ylabel('original image', rotation=90, loc='center', labelpad=10)
    imshow_task23(train_image[0, :, :, :].detach(), axs1[0])
    imshow_task23(test_image[0, :, :, :].detach(), axs2[0])
    for i in range(6):
        conv_per_channel = conv1_train.clone()
        conv_per_channel[0][:i] = 0
        conv_per_channel[0][i+1:] = 0
        deconv = net2.deconv2(F.relu(net2.unpool(conv_per_channel, indices1_train)))
        axs1[i + 1].set_ylabel(f'channel {i+1}', rotation=90, loc='center', labelpad=10)
        imshow_task23(deconv[0, :, :, :].detach(), axs1[i+1])
        conv_per_channel = conv1_test.clone()
        conv_per_channel[0][:i] = 0
        conv_per_channel[0][i + 1:] = 0
        deconv = net2.deconv2(F.relu(net2.unpool(conv_per_channel, indices1_test)))
        axs2[i + 1].set_ylabel(f'channel {i+1}', rotation=90, loc='center', labelpad=10)
        imshow_task23(deconv[0, :, :, :].detach(), axs2[i+1])
    fig3, axs3 = plt.subplots(4, 1, figsize=(10, 10))
    fig4, axs4 = plt.subplots(4, 1, figsize=(10, 10))
    conv2_train, indices2_train = net2.pool(F.relu(net2.conv2(conv1_train)))
    conv2_test, indices2_test = net2.pool(F.relu(net2.conv2(conv1_test)))
    fig3.suptitle('Train_image - second convolutional layer')
    fig4.suptitle('Test_image - second convolutional layer')
    axs3[0].set_ylabel('original image', rotation=90, loc='center', labelpad=10)
    axs4[0].set_ylabel('original image', rotation=90, loc='center', labelpad=10)
    imshow_task23(train_image[0, :, :, :].detach(), axs3[0])
    imshow_task23(test_image[0, :, :, :].detach(), axs4[0])
    for i in range(3):
        conv_per_channel = conv2_train.clone()
        conv_per_channel[0][:i] = 0
        conv_per_channel[0][i + 1:] = 0
        deconv = net2.deconv1(F.relu(net2.unpool(conv_per_channel, indices2_train)))
        deconv = net2.deconv2(F.relu(net2.unpool(deconv, indices1_train)))
        axs3[i + 1].set_ylabel(f'channel {i+1}', rotation=90, loc='center', labelpad=10)
        imshow_task23(deconv[0, :, :, :].detach(), axs3[i + 1])
        conv_per_channel = conv2_test.clone()
        conv_per_channel[0][:i] = 0
        conv_per_channel[0][i + 1:] = 0
        deconv = net2.deconv1(F.relu(net2.unpool(conv_per_channel, indices2_test)))
        deconv = net2.deconv2(F.relu(net2.unpool(deconv, indices1_test)))
        axs4[i + 1].set_ylabel(f'channel {i+1}', rotation=90, loc='center', labelpad=10)
        imshow_task23(deconv[0, :, :, :].detach(), axs4[i + 1])
    plt.show()


def main():
    task1()
    task2()
    task3()


if __name__ == '__main__':
    main()
