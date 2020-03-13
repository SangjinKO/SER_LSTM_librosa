import numpy as np
from keras.utils import np_utils
import Librosa_Feature #Feature Extraction
from DNN_Model import LSTM_Model
from Config import Config #File Path

print ("LSTM using librosa for RAVDESS")
print ("FEATURE: 21 features including MFCC, pitch, magnitude, etc")
print ("NOTE: change the name of file(corpus) folders to make its length less than 10")

def Train(save_model_name: str):
    Config.save_model_name = save_model_name
    x_train, x_test, y_train, y_test = Librosa_Feature.get_data(Config.DATA_PATH, Config.TRAIN_FEATURE_PATH_LIBROSA, train=True)

    # vectorize
    print(y_train)

    y_train = np_utils.to_categorical(y_train)
    y_val = np_utils.to_categorical(y_test)

    model = LSTM_Model(input_shape=x_train.shape[1], num_classes=len(Config.CLASS_LABELS))

    # 2D --> 3D (samples, time_steps, features)
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

    model.train(x_train, y_train, x_test, y_val, n_epochs = Config.epochs)
    print("END Training")
    #model.evaluate(x_test, y_test) #Need to check out
    model.save_model(save_model_name)

    return model


## Trainig & Validating
Train("LSTM_LIBROSA")
