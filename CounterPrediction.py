import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
# Libraries
import datetime

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, homogeneity_score, adjusted_rand_score, \
    roc_auc_score, roc_curve, f1_score, auc, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import v_measure_score
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import Perceptron
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import LabelPropagation
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.gaussian_process.gpc import GaussianProcessClassifier
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.linear_model.ridge import RidgeClassifierCV
from sklearn.neighbors.classification import RadiusNeighborsClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,homogeneity_score,adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import v_measure_score
from sklearn import metrics
from sklearn.svm import NuSVC
import time
from sklearn.svm.classes import OneClassSVM
from sklearn.naive_bayes import MultinomialNB
from datetime import datetime
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve



#To ignore warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import tensorflow.keras.layers as Layers
import tensorflow.keras.activations as Actications
import tensorflow.keras.models as Models
import tensorflow.keras.optimizers as Optimizer
import tensorflow.keras.metrics as Metrics
import tensorflow.keras.utils as Utils
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from keras.utils.vis_utils import model_to_dot
import os
import matplotlib.pyplot as plot
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as CM, mean_absolute_error
from random import randint
from IPython.display import SVG
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

import numpy as np                          # linear algebra
import os                                   # used for loading the data
from sklearn.metrics import confusion_matrix# confusion matrix to carry out error analysis
import seaborn as sn                        # heatmap
from sklearn.utils import shuffle           # shuffle the data
import matplotlib.pyplot as plt             # 2D plotting library
import cv2                                  # image processing library
import tensorflow as tf                     # best library ever
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plot





# Here's our 6 categories that we have to classify.
class_names = ['NORMAL','PNEUMONIA']
class_names_label = {'NORMAL': 0,
                    'PNEUMONIA' : 1,
                     }
nb_classes = 3


def load_data():
    """
        Load the data:
            - 14,034 images to train the network.
            - 10,000 images to evaluate how accurately the network learned to classify images.
    """

    datasets = ['/train',
                '/test']
    size = (150, 150)
    output = []
    for dataset in datasets:
        directory = "/home/peter/Desktop/Desktop/Peter T. Habib/Artificial_Intelligance/Finished/PneumonAI/chest_xray" + dataset
        images = []
        labels = []
        for folder in os.listdir(directory):
            curr_label = class_names_label[folder]
            for file in os.listdir(directory + "/" + folder):
                img_path = directory + "/" + folder + "/" + file
                curr_img = cv2.imread(img_path)
                curr_img = cv2.resize(curr_img, size)
                images.append(curr_img)
                labels.append(curr_label)
        images, labels = shuffle(images, labels)  ### Shuffle the data !!!
        images = np.array(images, dtype='float32')  ### Our images
        labels = np.array(labels, dtype='int32')  ### From 0 to num_classes-1!

        output.append((images, labels))

    return output





def get_classlabel(class_code):
    labels = {0:'NORMAL',1:'PNEUMONIA'}

    return labels[class_code]

(train_images, train_labels), (test_images, test_labels) = load_data()


test_images = test_images / 255.0



model_file_name = 'model_84%.h5'
model = load_model(model_file_name)

# test_loss = model.evaluate(test_images, test_labels)
# print('Model Accuracy: ',round(int(test_loss[1]*100)),'%')

SVG(model_to_dot(model).create(prog='dot', format='svg'))
Utils.plot_model(model,to_file='model.png',show_shapes=True)




for i in range(3):
    index = np.random.randint(test_images.shape[0]) # We choose a random index

    img = (np.expand_dims(test_images[index], 0))
    predictions = model.predict(img)     # Vector of probabilities
    pred_img = np.argmax(predictions[0]) # We take the highest probability
    pred_label = class_names[pred_img]
    true_label = class_names[test_labels[index]]
    title = 'Predicted: {} ' \
            'True Val : {}  '.format(pred_label,true_label)
    plt.figure()
    plt.imshow(test_images[index])
    plt.grid(False)
    plt.title(title)
    plt.show()




test_loss = model.evaluate(test_images, test_labels)
print('Model Accuracy: ',round(int(test_loss[1]*100)),'%')


print('Test loss:', test_loss[0])
print('Test accuracy:', test_loss[1])







