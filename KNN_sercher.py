from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.pairwise import cosine_similarity

import torch
from glob import glob
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class FolderDataset(Dataset):
    """Класс Dataset для загрузки изображений из папки. Нужен для создания итератора"""
    def __init__(self, folder_path, transform=None):
        self.image_paths = glob(os.path.join(folder_path, '*'))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, idx


def extract_features(data, transform):
    """Функция для извлечения признаков из изображений.
    Если data - путь к изображению, возвращает np.array с фичами этого изображения
    Если data - подготовленный DataLoader, возвращает np.array с фичами всего DataLoader"""
    # Преобразования для подготовки изображений
    # Загрузка и подготовка модели
    model = models.resnet50(pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    features = []

    if isinstance(data, str):
        # Один путь к изображению
        image = Image.open(data).convert('RGB')
        image = transform(image).unsqueeze(0)  # Добавление размерности батча
        with torch.no_grad():
            feature = model(image).squeeze().numpy()
        features.append(feature)

    if isinstance(data, np.ndarray):
        # Одно изображение
        transform = transforms.Compose([
            transforms.ToPILImage(),  # Преобразование numpy.ndarray в PIL.Image
            transform
        ])
        image = transform(data).unsqueeze(0)  # Добавление размерности батча
        with torch.no_grad():
            feature = model(image).squeeze().numpy()
        features.append(feature)

    elif isinstance(data, DataLoader):
        # Батч изображений
        for images, _ in data:
            images = images.to(next(model.parameters()).device)  # Перемещение изображений на GPU, если доступно
            with torch.no_grad():
                batch_features = model(images)
                batch_features = batch_features.cpu().numpy()  # Перемещение обратно на CPU
                features.extend(batch_features)

    else:
        raise TypeError("На входе должны быть или путь к файлу или DataLoader.")

    return np.array(features)


def create_features(input_folder: str, output_folder: str, transform):
    """Функция для создания файла с фичами. Алгоритм проходит по папке input_folder,
    сохраняет np.array с фичами в одноименный файл в папку output_folder.
    Args:
        input_folder (str): Путь к папке с изображениями
        output_folder (str): Путь к папке с фичами"""
    print("Начало подготовки признаков изображений")
    folders = list()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for root, dirs, files in os.walk(input_folder):
        # Определяем относительный путь к текущей папке относительно input_folder
        relative_path = os.path.relpath(root, input_folder)
        if relative_path != ".":
            output_subfolder = os.path.join(input_folder, relative_path)
            folders.append((output_subfolder, relative_path))
    # Загрузка изображений из папки и извлечение признаков
    step = 0
    steps = len(folders)
    for image_folder_path, name in folders:

        dataset = FolderDataset(image_folder_path, transform=transform)
        data_loader = DataLoader(dataset, batch_size=8, shuffle=False)

        # Извлечение признаков для всего датасета
        dataset_features = extract_features(data_loader, transform)
        np.save(os.path.join(output_folder, f"{name}.npy"), dataset_features)
        step += 1
        print(f"Шаг {step} из {steps} завершён")
    print("Признаки изображений сохранены")


def find_similar_images(new_image_features, dataset_features, k=1):
    """Функция для поиска максимально похожих изображений
    Args:
        new_image_features (np.array): фичи базового изображения
        dataset_features (np.array): массив с фичами, по которым ищем
        k (int): кол-во искомых изображений
    return:
        similar_indices: индексы подходящих изображений
        similarity_scores: похожесть
    """

    # используем косинусное сходство
    similarities = cosine_similarity(new_image_features.reshape(1, -1), dataset_features)
    similarity_scores = similarities[0]
    similar_indices = np.argsort(-similarity_scores)[:k]
    return similar_indices, similarity_scores


def show_image(image_path):
    """Функция для отображения изображения по пути"""
    image = Image.open(image_path)
    plt.imshow(image)
    plt.show()


def extract_feature_and_find_similar(new_image_path, input_class, dataset_path, features_folder_path, k, transform):
    """Функция для создания фичей входного изображения и поиска максимально похожих
    Args:
        new_image_path (str): Путь к изображению
        input_class (str): класс, в котором надо искать
        dataset_path (str): путь к папке с изображениями
        features_folder_path (str): путь к папке с фичами
        k (int): кол-во искомых изображений
    """
    new_image_features = extract_features(new_image_path, transform)
    dataset_features = np.load(os.path.join(features_folder_path, f'{input_class}.npy'))
    similar_images_indices, similarity_scores = find_similar_images(new_image_features, dataset_features, k)
    print(input_class, os.path.join(dataset_path, input_class))
    dataset = FolderDataset(os.path.join(dataset_path, input_class),  transform=transform)
    dataset_image_paths = [dataset.image_paths[i] for i in similar_images_indices]
    print(f"Индексы наиболее похожих изображений: {similar_images_indices}")
    print(f"Оценки сходства: {similarity_scores[similar_images_indices]}")
    return dataset_image_paths
