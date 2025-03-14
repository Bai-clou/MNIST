import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class FCNet(nn.Module):
    def __init__(self, input_size=784, num_classes=10, depth=3, width=256, activation='relu'):
        super(FCNet, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(depth - 1):
            in_features = input_size if i == 0 else width
            self.layers.append(nn.Linear(in_features, width))

            if activation == 'relu':
                self.layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                self.layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                self.layers.append(nn.Tanh())
            else:
                raise ValueError("Unsupported activation function")

        self.output = nn.Linear(width, num_classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


def load_data(batch_size=64):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = torchvision.datasets.MNIST(
        root='./data_set', train=True, download=False, transform=transform)
    test_set = torchvision.datasets.MNIST(
        root='./data_set', train=False, download=False, transform=transform)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(test_set, batch_size=1000, shuffle=False)
    )


def plot_results(results, config):
    plt.figure(figsize=(12, 5))

    # 绘制训练损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(results['train_loss'], 'b-o', label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    # 绘制测试准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(results['test_acc'], 'r-s', label='Test Accuracy')
    plt.title('Test Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    # 根据实际准确率范围动态调整
    min_acc = min(results['test_acc']) - 1  # 最小值减1，留出一些空间
    max_acc = max(results['test_acc']) + 1  # 最大值加1，留出一些空间
    plt.ylim(min_acc, max_acc)  # 动态设置纵轴范围
    plt.grid(True)

    # 显示配置参数
    config_str = '\n'.join([f'{k}: {v}' for k, v in config.items()])
    plt.suptitle(f"Model Configuration:\n{config_str}", y=1.05)
    plt.tight_layout()

    # 保存和显示图像
    plt.savefig(f"new_results_{config['depth']}layers_{config['activation']}_{config['lr']}lr.png",
                bbox_inches='tight')
    plt.show()


def train(model, device, train_loader, test_loader, optimizer, scheduler=None, epochs=10):
    criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'test_acc': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if scheduler:
            scheduler.step()

        avg_loss = running_loss / len(train_loader)
        results['train_loss'].append(avg_loss)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        acc = 100 * correct / total
        results['test_acc'].append(acc)

        print(f'Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {acc:.2f}%')

    return results


def main():
    device = torch.device("cpu")
    batch_size = 64
    epochs = 15

    experiments = [
        {'depth': 3, 'width': 256, 'activation': 'relu', 'lr': 0.001},
        # {'depth': 5, 'width': 512, 'activation': 'tanh', 'lr': 0.0005},
        # {'depth': 4, 'width': 256, 'activation': 'sigmoid', 'lr': 0.005}
    ]

    for exp in experiments:
        print(f"\nRunning experiment with config: {exp}")
        train_loader, test_loader = load_data(batch_size)

        model = FCNet(
            depth=exp['depth'],
            width=exp['width'],
            activation=exp['activation']
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=exp['lr'])
        results = train(model, device, train_loader, test_loader, optimizer, epochs=epochs)

        # 可视化结果
        plot_results(results, exp)


if __name__ == '__main__':
    main()