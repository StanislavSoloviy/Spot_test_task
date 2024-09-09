from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
import os
import shutil


def create_and_split_dataset(dataset_path : str,
                             valid_ratio: float = 0.1,
                             test_ratio: float = 0.1,
                             batch_size: int = 16
                             ):
    """Функция для создания датасета, возвращает кортеж из (train_loader, valid_loader, test_loader)

    Args:
        dataset_path (str): Путь к исходной папке с изображениями.
        valid_ratio (float): Процент датасета на валидацию
        test_ratio (float): Процент датасета на тестирование
        batch_size (int): Размер батча
    """
    # Трансформации для предварительной обработки данных
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Изменение размера изображений
        transforms.ToTensor(),  # Преобразование в тензор
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация
    ])
    """Функция для подготовки датасета, возвращает кортеж из (train_loader, valid_loader, test_loader)"""
    # Загрузка полного датасета
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    # Получение длины каждого поднабора
    dataset_size = len(dataset)
    valid_size = int(valid_ratio * dataset_size)
    test_size = int(test_ratio * dataset_size)
    train_size = dataset_size - test_size - valid_size  # остаток данных для теста

    # Разделение датасета
    train_data, valid_data, test_data = random_split(dataset, [train_size, valid_size, test_size])
    class_names = train_data.dataset.classes  # Извлекаем названия классов

    # Создание DataLoader'ов
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print("Подготовка датасета завершена")
    print(f'Размеры наборов данных: Train={len(train_data)}, Valid={len(valid_data)}, Test={len(test_data)}')
    return class_names, train_loader, valid_loader, test_loader


def augment_dataset(input_folder, output_folder, augment_count=5, balance=False):
    """
    Функция для рекурсивной аугментации изображений в папках и вложенных папках.

    Args:
        input_folder (str): Путь к исходной папке с изображениями.
        output_folder (str): Путь к папке, куда будут сохранены аугментированные изображения.
        augment_count (int): Количество аугментаций на одно изображение.
        balance (bool): балансировать классы? True - да / False - нет
    """
    print("Начало аугментации")
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    # Трансформации для аугментации
    augmentation_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # Случайное горизонтальное отражение
        transforms.RandomRotation(degrees=30),  # Случайный поворот на ±30 градусов
        transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0)),  # Случайный кроп
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # Изменение яркости, контраста, насыщенности и оттенка
        transforms.RandomGrayscale(p=0.1),  # Случайное преобразование в черно-белое
        transforms.ToTensor()  # Преобразование в тензор
    ])
    # Объявляем max_count, далее присвоим ему значение
    max_count = 0
    # Если надо баласировать классы, то в каждом классе будет макс. кол-во изображений, умноженное на augment_count
    if balance:
        # Словарь для подсчета количества изображений в каждом классе
        class_counts = {}

        # Рекурсивно проходим по каждой папке и файлам в input_folder
        for root, dirs, files in os.walk(input_folder):
            # Определяем относительный путь к текущей папке относительно input_folder
            relative_path = os.path.relpath(root, input_folder)

            # Проверяем, есть ли уже эта папка в class_counts
            if relative_path not in class_counts:
                class_counts[relative_path] = 0

            # Подсчитываем количество изображений в каждом классе
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):  # Обрабатываем только изображения
                    class_counts[relative_path] += 1

        # Определяем максимальное количество изображений в одном классе
        max_count = max(class_counts.values()) * augment_count
    # Рекурсивно проходим по каждой папке и файлам в input_folder
    for root, dirs, files in os.walk(input_folder):
        # Определяем относительный путь к текущей папке относительно input_folder
        relative_path = os.path.relpath(root, input_folder)

        # Определяем соответствующую папку в output_folder
        output_subfolder = os.path.join(output_folder, relative_path)

        # Если выходной подкаталог не существует, создаем его
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        if balance and len(files) > 0:
            current_augment_count = int(max_count / len(files))
        else:
            current_augment_count = augment_count

        # Обрабатываем изображения в текущей папке
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):  # Обрабатываем только изображения
                image_path = os.path.join(root, file)
                image = Image.open(image_path).convert('RGB')

                # Применяем аугментацию augment_count раз для каждого изображения
                for i in range(current_augment_count):
                    augmented_image = augmentation_transforms(image)  # Применение трансформаций
                    augmented_image = transforms.ToPILImage()(
                        augmented_image)  # Преобразование обратно в изображение PIL

                    # Генерируем уникальное имя для аугментированного изображения
                    base_name = os.path.splitext(file)[0]
                    new_file_name = f"{base_name}_aug_{i + 1}.jpg"
                    output_path = os.path.join(output_subfolder, new_file_name)

                    # Сохраняем аугментированное изображение
                    augmented_image.save(output_path)

    print(f"Процесс аугментации завершен. Изображения сохранены в: {output_folder}")


def prepare_dataset(input_folder: str,
                    augmentation: bool = True,
                    balance: bool = False,
                    augment_count: int= 2,
                    valid_ratio: float = 0.1,
                    test_ratio: float = 0.1,
                    batch_size: int = 16
                    ):
    """Функция для подготовки датасета, возвращает кортеж из (train_loader, valid_loader, test_loader)

    Args:
        input_folder (str): Путь к исходной папке с изображениями.
        augmentation (bool) Нужно ли аугментировать датасет?
        balance (bool): балансировать классы? True - да / False - нет
        augment_count (int): Количество аугментаций на одно изображение.
        valid_ratio (float): Процент датасета на валидацию
        test_ratio (float): Процент датасета на тестирование
        batch_size (int): Размер батча
    """

    print("Начало подготовки датасета")
    # Если нужно аугментировать датасет - делаем это
    dataset_path = input_folder # определяем путь для датасета
    # Если выходной подкаталог не существует выводим ошибку
    if not os.path.exists(dataset_path):
        print("Указанная папка отсутствует")
        return None, None, None, None
    else:
        if augmentation:
            dataset_path += "_aug"
            augment_dataset(input_folder=input_folder,
                            output_folder=dataset_path,
                            balance=balance,
                            augment_count=augment_count
                            )

        # Создаём train/valid/test датасеты
        return create_and_split_dataset(dataset_path=dataset_path,
                                        valid_ratio=valid_ratio,
                                        test_ratio=test_ratio,
                                        batch_size=batch_size
                                        )



