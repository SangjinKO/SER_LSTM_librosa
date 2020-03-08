
import matplotlib.pyplot as plt
from keras.models import model_from_json
from sklearn.externals import joblib


def load_model(load_model_name: str, model_name: str):

    if(model_name == 'lstm'):

        model_path = 'Models/' + load_model_name + '.h5'
        model_json_path = 'Models/' + load_model_name + '.json'
        
        json_file = open(model_json_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        model.load_weights(model_path)
    
    elif(model_name == 'svm' or model_name == 'mlp'):
        model_path = 'Models/' + load_model_name + '.m'
        model = joblib.load(model_path)

    return model

def plotCurve(train, val, title: str, y_label: str):
    plt.plot(train)
    plt.plot(val)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
