import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from torchvision import transforms
from PIL import Image
import json


class CNNClassifier(nn.Module):
    """Класс нафикатора на базе свёрточной нейросети"""
    def __init__(self, num_classes):
        """Инициализация модели"""
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.class_names = list()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)  # Разворачиваем тензор
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def train_me(self, train_loader, valid_loader, criterion, num_epochs, lr, class_names):
        """Функция для обучения модели

        Args:
            train_loader (DataLoader): тренировочный датасет
            valid_loader (DataLoader): валидационный датасет
            criterion: функция потерь
            num_epochs (int): кол-во эпох для обучения
            lr (float): Коэффициент скорости обучения
            class_names (list): именна классов
            """
        self.class_names = class_names
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()  # Установить режим обучения
        for epoch in range(num_epochs):
            running_loss = 0.0
            self.train()  # Переводим модель в режим обучения

            # Обучение на тренировочных данных
            for images, labels in train_loader:
                optimizer.zero_grad()  # Обнуление градиентов

                outputs = self(images)  # Прямой проход
                loss = criterion(outputs, labels)  # Вычисление ошибки
                loss.backward()  # Обратный проход
                optimizer.step()  # Оптимизация

                running_loss += loss.item()

            # Переход на валидационные данные
            self.eval()  # Переводим модель в режим оценки
            valid_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():  # Отключаем вычисление градиентов
                for images, labels in valid_loader:
                    outputs = self(images)
                    loss = criterion(outputs, labels)
                    valid_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            valid_accuracy = 100 * correct / total

            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Train Loss: {running_loss / len(train_loader):.4f}, "
                  f"Valid Loss: {valid_loss / len(valid_loader):.4f}, "
                  f"Valid Accuracy: {valid_accuracy:.2f}%")

        print('Обучение завершено')

    def test(self, test_loader):
        """Функция для тестирования модели
        Args:
            test_loader (DataLoader): тестовый датасет
            """
        self.eval()  # Переводим модель в режим оценки
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())  # Сохраняем истинные значения
                all_predictions.extend(predicted.cpu().numpy())  # Сохраняем предсказания

        # Вычисляем метрики
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')

        print(f'Accuracy: {accuracy * 100:.2f}%')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')

        # Вычисляем матрицу ошибок
        cm = confusion_matrix(all_labels, all_predictions)

        # Визуализация матрицы ошибок
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

    def predict_image(self, image_path: str) -> str:
        """Функция для предсказания результатов модели
        Args:
            image_path (str): путь до картинки на компьютере
        return:
            название класса предсказания
            """
        # Трансформации должны быть такими же, как при обучении
        transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Изменение размера изображения на 128x128
            transforms.ToTensor(),  # Преобразование изображения в тензор
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация
        ])

        # Открываем изображение и применяем трансформации
        image = Image.open(image_path).convert('RGB')  # Открываем изображение в RGB
        image = transform(image)  # Применяем преобразования
        image = image.unsqueeze(0)  # Добавляем batch размер (1, C, H, W)

        # Переводим модель в режим оценки
        self.eval()

        # Прогоняем изображение через модель
        with torch.no_grad():
            output = self(image)
            _, predicted = torch.max(output.data, 1)

        # Получаем предсказанный класс
        predicted_class = self.class_names[predicted.item()]

        print(f"Предсказанный класс: {predicted_class}")
        return predicted_class

    def save(self, path_weiths: str, path_class_names: str):
        """Функция для сохранения весов обученной модели
        Args:
            path_weiths (str): путь к файлу с весами
            path_class_names (str): путь к файлу с именами классов
            """
        try:
            torch.save(self.state_dict(), path_weiths)
            with open(path_class_names, "w", encoding="UTF-8") as file:
                json.dump(self.class_names, file)
            print("Веса модели успешно сохранены в файл ", path_weiths)
        except Exception as err:
            print(err)

    def load(self, path_weiths, path_class_names):
        """Функция для загрузки весов обученной ранее модели
        Args:
            path_weiths (str): путь к файлу с весами
            path_class_names (str): путь к файлу с именами классов
            """
        try:
            self.load_state_dict(torch.load(path_weiths))
            with open(path_class_names, "r", encoding="UTF-8") as file:
                self.class_names = json.load(file)
            print("Веса модели успешно загружены из файла ", path_weiths)
        except Exception as err:
            print(err)
