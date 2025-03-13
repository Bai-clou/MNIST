import torch
from torch import nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 定义模型结构
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)  # 展平
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 加载模型
model = SimpleCNN()
model.load_state_dict(torch.load('mnist_cnn.pth'))
model.eval()

# 加载并处理手写数字图片
image = Image.open('test2.png').convert('L')
image = image.resize((28, 28))
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
image_tensor = transform(image).unsqueeze(0)

# 推理
with torch.no_grad():
    output = model(image_tensor)
_, predicted = torch.max(output, 1)
predicted_label = predicted.item()

# 可视化结果
plt.imshow(image_tensor.squeeze(), cmap='gray')
plt.title(f'Predicted: {predicted_label}')
plt.axis('off')
plt.show()

print(f'Predicted Label: {predicted_label}')