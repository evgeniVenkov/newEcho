import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 1. Определяем архитектуру сети
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 256, 1)  # Входной слой
        self.sigmoid = nn.Sigmoid()          # Активация для бинарной классификации

    def forward(self, x):
        x = self.flatten(x)
        x = self.sigmoid(x)
        return x

# 2. Заготовка датасета
class SymbolDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images  # Список изображений
        self.labels = labels  # Соответствующие метки (0 или 1)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# 3. Алгоритм обучения

def train_model(model, dataloader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Обнуление градиентов
            optimizer.zero_grad()

            # Прямой проход
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)

            # Обратный проход и оптимизация
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

# 4. Устройство для вычислений
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5. Генерация случайных данных для теста
# Пока вместо картинок генерируем случайные данные
num_samples = 1000
input_size = (128, 256)

fake_images = np.random.rand(num_samples, *input_size).astype(np.float32)
fake_labels = np.random.randint(0, 2, num_samples)

transform = transforms.Compose([
    transforms.ToTensor()
])

# 6. Датасет и загрузчик
train_dataset = SymbolDataset(fake_images, fake_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 7. Инициализация модели, функции потерь и оптимизатора
model = BinaryClassifier().to(device)
criterion = nn.BCELoss()  # Бинарная кросс-энтропия
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 8. Обучение модели
train_model(model, train_loader, criterion, optimizer, num_epochs=10)
