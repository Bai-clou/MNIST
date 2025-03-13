import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F

# super parameters
batch_size = 64
learning_rate = 0.01
momentum = 0.5
EPOCH = 10

#归一化
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

#载入数据集
train_dataset = datasets.MNIST(root='./data_set', train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root='./data_set', train=False, download=False, transform=transform)

#载入数据集，打乱训练集
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# fig = plt.figure()
# for i in range(12):
#     plt.subplot(3, 4, i+1)
#     plt.tight_layout()
#     plt.imshow(train_dataset.train_data[i], cmap='gray', interpolation='none')
#     plt.title("Labels: {}".format(train_dataset.train_labels[i]))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()


