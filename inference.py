from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from model import get_model
from utils import load_model

# Указываем количество классов
num_classes = 20

# Классы пыльцы (из class_map)
class_names = [
    'buckwheat', 'clover', 'angelica', 'angelica_garden', 'willow', 'hill_mustard',
    'linden', 'meadow_pink', 'alder', 'birch', 'fireweed', 'nettle', 'pigweed',
    'plantain', 'sorrel', 'grass', 'pine', 'maple', 'hazel', 'mugwort'
]

# Создаем и загружаем модель
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model(num_classes)
model = load_model(model, 'faster_rcnn_pollen_model.pth')
model.to(device)
model.eval()


def run_inference(image_path, model):
    image = Image.open(image_path).convert("RGB")
    transform = T.ToTensor()
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(image)

    return predictions[0]


def visualize_predictions(image_path, predictions, confidence_threshold=0.5):
    image = Image.open(image_path)
    plt.imshow(image)

    # Фильтрация по порогу уверенности
    high_conf_preds = []
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score >= confidence_threshold:
            high_conf_preds.append((box, label, score))

    if not high_conf_preds:
        print(f"No predictions with confidence >= {confidence_threshold}")
        return

    # Выводим предсказанные боксы и классы
    for box, label, score in high_conf_preds:
        xmin, ymin, xmax, ymax = box
        class_name = class_names[label]
        print(f"Class: {class_name}, Score: {score:.2f}, Box: [{xmin}, {ymin}, {xmax}, {ymax}]")

        # Визуализация прямоугольников на изображении
        plt.gca().add_patch(
            plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='red', linewidth=2)
        )

        # Выводим название класса на изображении
        plt.text(xmin, ymin, f'{class_name}: {score:.2f}', color='white', fontsize=12, backgroundcolor='red')

    plt.show()


# Вставьте путь к вашему изображению
image_path = 'pollen20ldet/test.png'
predictions = run_inference(image_path, model)

# Визуализируем предсказания и выводим их в текстовом виде
visualize_predictions(image_path, predictions, confidence_threshold=0.1)
