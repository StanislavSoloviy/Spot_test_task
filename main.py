from torch import nn, optim


from Prepairing_Dataset import prepair_dataset
from HYPERPARAMETERS import DATASET_PATH, VALID_RATIO, TEST_RATIO, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, NUM_CLASSES, \
    MODEL_WEITHS_PATH, CLASS_NAMES_PATH
from CNN_Classifier import CNNClassifier
from KNN_sercher import extract_feature_and_find_similar


if __name__ == '__main__':
    # prepair_dataset(input_folder=DATASET_PATH,
    #                 augmentation=True,
    #                 balance=True,
    #                 augment_count=2,
    #                 valid_ratio=VALID_RATIO,
    #                 test_ratio=TEST_RATIO,
    #                 batch_size=BATCH_SIZE
    #                 )

    class_names, train_loader, valid_loader, test_loader = prepair_dataset(input_folder=DATASET_PATH+"_aug",
                                                              augmentation=False,
                                                              balance=False,
                                                              augment_count=2,
                                                              valid_ratio=VALID_RATIO,
                                                              test_ratio=TEST_RATIO,
                                                              batch_size=BATCH_SIZE
                                                              )
    criterion = nn.CrossEntropyLoss()
    model_classifier = CNNClassifier(num_classes=NUM_CLASSES)
    #model_classifier.train_me(train_loader, valid_loader, criterion, NUM_EPOCHS, LEARNING_RATE, class_names)
    #model_classifier.save(MODEL_WEITHS_PATH, CLASS_NAMES_PATH)
    model_classifier.load(MODEL_WEITHS_PATH, CLASS_NAMES_PATH)
    #model_classifier.test(test_loader)

    image_path = "./glass.jpg"
    image_class = model_classifier.predict_image(image_path)

    # Пример использования поиска
    k = 3
    extract_feature_and_find_similar(image_path, k, image_class)

