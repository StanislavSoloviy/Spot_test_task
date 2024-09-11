from HYPERPARAMETERS import VALID_RATIO, TEST_RATIO, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, \
    MODEL_WEIGHTS_PATH, CLASS_NAMES_PATH, FEATURES_FOLDER_PATH, SETTINGS_PATH
from CNN_Classifier import CNNClassifier
from Gradio import WebApplication
from Settings import load_dataset_path


if __name__ == '__main__':
    # Объявляем модель
    dataset_path = load_dataset_path(SETTINGS_PATH)
    model_classifier = CNNClassifier(lr=LEARNING_RATE)
    # Объявляем приложение Gradio
    wep_API = WebApplication(model_classifier=model_classifier,
                             model_weight_path=MODEL_WEIGHTS_PATH,
                             class_names_path=CLASS_NAMES_PATH,
                             dataset_path=dataset_path,
                             features_folder_path=FEATURES_FOLDER_PATH,
                             epoches=NUM_EPOCHS,
                             valid_ratio=VALID_RATIO,
                             test_ratio=TEST_RATIO,
                             batch_size=BATCH_SIZE,
                             )
    # Запускаем приложение Gradio
    wep_API.launch()
