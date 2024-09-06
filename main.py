import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import timm

# Параметры обучения
num_classes = 20  # Количество классов в вашем датасете
batch_size = 32
epochs = 10
learning_rate = 0.001

# Преобразования изображений (предобработка)
preprocess = transforms.Compose([
    transforms.Resize(331),  # Размер для NASNet
    transforms.CenterCrop(331),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# Определение датасета
class PollenDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)  # Загружаем CSV с метками классов
        self.root_dir = root_dir  # Путь к изображениям
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])  # Путь к изображению
        image = Image.open(img_name).convert('RGB')  # Открываем изображение
        label = int(self.annotations.iloc[idx, 1])  # Метка класса (предположим, что она числовая)

        if self.transform:
            image = self.transform(image)

        return image, label


# Загрузка данных
dataset = PollenDataset(csv_file='pollen20ldet/class_map.csv', root_dir='pollen20ldet/images', transform=preprocess)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Загрузка предобученной модели NASNet
model = timm.create_model('nasnetalarge', pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)  # Изменение выхода под количество классов
model = model.cuda()  # Если доступен GPU

# Оптимизатор и функция потерь
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Цикл обучения
model.train()  # Переводим модель в режим обучения

for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()  # Если используем GPU

        optimizer.zero_grad()  # Сбрасываем градиенты
        outputs = model(inputs)  # Прогоняем данные через модель
        loss = criterion(outputs, labels)  # Вычисляем ошибку
        loss.backward()  # Обратное распространение ошибки
        optimizer.step()  # Обновление весов

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader)}')

print("Обучение завершено!")
