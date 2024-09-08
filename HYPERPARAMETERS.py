# Гиперпараметры
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
NUM_CLASSES = 7  # 6 классов + класс "other"

# TRAIN_RATIO = 0.7   # 70% для обучения
VALID_RATIO = 0.15  # 15% для валидации
TEST_RATIO = 0.15   # 15% для тестирования

DATASET_PATH = './test_data'
MODEL_WEITHS_PATH = 'classifier_weiths.pth'
CLASS_NAMES_PATH = 'class_names.json'
FEATURES_FOLDER_PATH = "./features"
