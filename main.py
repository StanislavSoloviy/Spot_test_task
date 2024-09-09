from HYPERPARAMETERS import DATASET_PATH, VALID_RATIO, TEST_RATIO, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, NUM_CLASSES, \
    MODEL_WEIGHTS_PATH, CLASS_NAMES_PATH, FEATURES_FOLDER_PATH
from CNN_Classifier import CNNClassifier
from Gradio import WebApplication


if __name__ == '__main__':
    # Объявляем модель
    model_classifier = CNNClassifier(num_classes=NUM_CLASSES, lr=LEARNING_RATE)
    # Объявляем приложение Gradio
    wep_API = WebApplication(model_classifier=model_classifier,
                             num_classes=NUM_CLASSES,
                             model_weight_path=MODEL_WEIGHTS_PATH,
                             class_names_path=CLASS_NAMES_PATH,
                             dataset_path=DATASET_PATH,
                             features_folder_path=FEATURES_FOLDER_PATH,
                             epoches=NUM_EPOCHS,
                             valid_ratio=VALID_RATIO,
                             test_ratio=TEST_RATIO,
                             batch_size=BATCH_SIZE,
                             )
    # Запускаем приложение Gradio
    wep_API.launch()
