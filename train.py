import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from dataset import PollenDataset
from model import get_model
from utils import save_model

# Пример использования аугментаций
transforms = T.Compose([
    T.ToTensor(),
])

# Инициализация датасета и DataLoader
dataset = PollenDataset(csv_file='pollen20ldet/bboxes.csv', root_dir='pollen20ldet/images/', transforms=transforms)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Количество классов
num_classes = 20  # 19 классов пыльцы + 1 фон
model = get_model(num_classes)

# Используем GPU, если доступно
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Оптимизатор
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Тренировочный цикл
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    i = 0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch: {epoch}, Step: {i}, Loss: {losses.item()}")
        i += 1

# Сохраняем модель после тренировки
save_model(model, 'faster_rcnn_pollen_model.pth')
