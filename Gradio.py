import gradio as gr
import sys
import io
from PIL import Image
import torch

from GLOBAL import transform
from KNN_sercher import extract_feature_and_find_similar, create_features
from Preparing_Dataset import prepare_dataset


class WebApplication():
    """Класс приложения"""

    def __init__(self, model_classifier, num_classes, model_weight_path, class_names_path, dataset_path,
                 features_folder_path, epoches, valid_ratio, test_ratio, batch_size):
        self._HEIGHT = 600  # Параметр для CSS кастомизации
        self._WIDTH = 600  # Параметр для CSS кастомизации
        self._DELIMETER = 2.05  # Параметр для CSS кастомизации
        self._model_classifier = model_classifier  # модель CNN
        self._num_classes = num_classes  # кол-во классов
        self._model_weight_path = model_weight_path  # путь до файла с весами модели классификатора
        self._class_names_path = class_names_path  # путь до файла с именами классов
        self._dataset_path = dataset_path  # путь до папки датасета
        self._epoches = epoches  # кол-во эпох
        self._features_folder_path = features_folder_path  # путь до папки с фичами
        self._valid_ratio = valid_ratio  # процент валидации
        self._test_ratio = test_ratio  # процент тестирования
        self._batch_size = batch_size  # размер батча
        """Страница 1, на которой проискодит поиск изображений."""
        with gr.Blocks(
                css=".custom-button { width: " + str(self._WIDTH) + "px; height: 50px; font-size: 16px; }") as page1:
            with gr.Row():
                with gr.Column():
                    gr.Markdown('# Загрузите картинку и нажмите на кнопку "Начать"')
                    img = gr.Image(label="Загрузите картинку", height=self._HEIGHT, width=self._WIDTH)
                    btn = gr.Button(value="Начать", interactive=False, elem_classes=["custom-button"])
                with gr.Column():
                    gr.Markdown('# Похожие изображения')
                    with gr.Row():
                        result1 = gr.Image(height=int(self._HEIGHT / self._DELIMETER),
                                           width=int(self._WIDTH / self._DELIMETER),
                                           visible=True)
                        result2 = gr.Image(height=int(self._HEIGHT / self._DELIMETER),
                                           width=int(self._WIDTH / self._DELIMETER),
                                           visible=True)
                    with gr.Row():
                        result3 = gr.Image(height=int(self._HEIGHT / self._DELIMETER),
                                           width=int(self._WIDTH / self._DELIMETER),
                                           visible=True)
                        result4 = gr.Image(height=int(self._HEIGHT / self._DELIMETER),
                                           width=int(self._WIDTH / self._DELIMETER),
                                           visible=True)

            output = [result1, result2, result3, result4]
            btn.click(self.upload_image, inputs=img, outputs=output)

            img.upload(fn=self.set_interactive, inputs=img, outputs=btn)
            img.clear(fn=self.set_interactive, inputs=img, outputs=btn)

        """Страница 2, на которой обучаем модели"""
        with gr.Blocks() as page2:
            with gr.Row():
                with gr.Column():
                    gr.Markdown('# Выберите необходимые параметры')
                    aug = gr.Checkbox(label="Аугментировать датасет?"),
                    aug_val = gr.Slider(1, 10, value=4, step=1, label="Количество аугментаций")
                    balance = gr.Checkbox(label="Балансировать классы при аугментации?"),
                    path = gr.Textbox(value=self._dataset_path, label="Путь до датасета")
                    epoches = gr.Slider(1, 20, value=self._epoches, step=1, label="Количество эпох при обучении")
                    create_feat = gr.Checkbox(label="Создать признаки изображений?"),
                    btn = gr.Button(value="Начать обучение")
                    output = gr.Textbox(label="Консольный вывод", lines=10)

            btn.click(self.start_train_models, inputs=[aug[0], aug_val, balance[0], path, epoches, create_feat[0]],
                      outputs=None)
            page2.load(self.update_console, outputs=output, every=0.5)

        self.app = gr.TabbedInterface([page1, page2], ["Поиск изображений", "Обучение модели"])
        self.console_output = self.ConsoleOutput()

    def launch(self):
        """Запуск приложения"""
        self.app.launch()

    class ConsoleOutput(io.StringIO):
        """Класс для передачи текста из консоли"""

        def __init__(self):
            super().__init__()

        def write(self, s):
            super().write(s)
            self.flush()

        def flush(self):
            pass

    def start_console(self):
        """Метод запуска консоли и обновления вывода"""
        sys.stdout = self.console_output  # Перенаправляем stdout в наш буфер
        # Возвращаем накопленный вывод
        return self.console_output.getvalue()

    def update_console(self):
        """Возвращает текущий накопленный вывод"""
        return self.console_output.getvalue()

    def upload_image(self, input_img):
        """Загрузка изображения"""
        self._model_classifier.load(self._model_weight_path, self._class_names_path)
        image_class = self._model_classifier.predict_image(input_img)
        if image_class in self._model_classifier.class_names:
            images_paths = extract_feature_and_find_similar(input_img, image_class, self._dataset_path,
                                                            self._features_folder_path, 4, transform)
            result = [Image.open(path).convert('RGB') for path in images_paths]
            return result
        else:
            return None

    @staticmethod
    def set_interactive(img):
        return gr.update(interactive=img is not None)

    def start_train_models(self, aug, aug_val, balance, path, epoches, create_feats):
        """Метод обучения модели. Принимаем данные с веб-клиента
        Args:
            aug (bool): нужно ли аугментировать
            aug_val (int): кол-во аугментаций
            balance (bool): Нужно ли балансировать классы
            path (str): Путь до датасета
            epoches (int): кол-во Эпох при обучении
            create_feats (bool): Создавать ли признаки для модели поиска изображений
        """
        self.start_console()
        # Создание датасета
        class_names, train_loader, valid_loader, test_loader = prepare_dataset(input_folder=path,
                                                                               augmentation=aug,
                                                                               balance=balance,
                                                                               augment_count=aug_val,
                                                                               valid_ratio=self._valid_ratio,
                                                                               test_ratio=self._test_ratio,
                                                                               batch_size=self._batch_size
                                                                               )
        # Обучение модели
        if class_names:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model_classifier.to(device)
            self._model_classifier.train_me(train_loader, valid_loader, epoches, class_names)
            self._model_classifier.save(self._model_weight_path, self._class_names_path)
            # Создание признаков
            if create_feats:
                create_features(input_folder=path,
                                output_folder=self._features_folder_path,
                                transform=transform
                                )
            print("Модель полностью готова к работе. Перейдите во вкладку <<Поиск изображения>> ")
        return None
