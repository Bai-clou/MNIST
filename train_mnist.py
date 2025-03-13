import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)

# 定义更深的模型
class DeeperCNN(nn.Module):
    def __init__(self, dropout_prob=0.3, stochastic_depth_prob=0.2):
        super(DeeperCNN, self).__init__()
        self.dropout_prob = dropout_prob
        self.stochastic_depth_prob = stochastic_depth_prob

        # 卷积层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # 全连接层
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 激活函数
        self.relu = nn.ReLU()

        # Dropout 层
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # Stochastic Depth: 随机跳过 conv1
        if self.training and torch.rand(1).item() < self.stochastic_depth_prob:
            x = self.pool(self.relu(self.bn1(self.conv1(x))))
        else:
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)

        # 第二层卷积
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # 第三层卷积
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        # 展平
        x = x.view(-1, 256 * 3 * 3)

        # 全连接层
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# 加载 MNIST 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data_set', train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root='./data_set', train=False, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 训练函数
def train_model(dropout_prob, stochastic_depth_prob, weight_decay):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeeperCNN(dropout_prob=dropout_prob, stochastic_depth_prob=stochastic_depth_prob).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=weight_decay)

    train_losses, test_losses, accuracies = [], [], []

    for epoch in range(15):  # 训练 15 个 epoch
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # 测试
        model.eval()
        test_loss, accuracy = 0.0, 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                accuracy += (predicted == labels).sum().item()
        test_loss /= len(test_loader)
        accuracy /= len(test_dataset)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')

    return train_losses, test_losses, accuracies

# 训练不同正则化方法的模型
results = {
    'Baseline': train_model(dropout_prob=0.0, stochastic_depth_prob=0.0, weight_decay=0.0),
    'Weight Decay': train_model(dropout_prob=0.0, stochastic_depth_prob=0.0, weight_decay=0.001),
    'Dropout': train_model(dropout_prob=0.3, stochastic_depth_prob=0.0, weight_decay=0.0),
    'Stochastic Depth': train_model(dropout_prob=0.0, stochastic_depth_prob=0.2, weight_decay=0.0),
}

# 可视化结果
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
for label, (train_losses, _, _) in results.items():
    plt.plot(train_losses, label=label)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
for label, (_, test_losses, _) in results.items():
    plt.plot(test_losses, label=label)
plt.title('Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 3)
for label, (_, _, accuracies) in results.items():
    plt.plot(accuracies, label=label)
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()