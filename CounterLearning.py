import warnings

from keras import Sequential
from keras.layers import Conv2D, LeakyReLU, MaxPooling2D, Dropout, Flatten, Dense

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
import sys
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
from sklearn.metrics import confusion_matrix as CM
from random import randint
from IPython.display import SVG
import matplotlib.gridspec as gridspec




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



(train_images, train_labels), (test_images, test_labels) = load_data()



print ("Number of training examples: " + str(train_labels.shape[0]))
print ("Number of testing examples: " + str(test_labels.shape[0]))
print ("Each image is of size: " + str(train_images.shape[1:]))





train_images = train_images / 255.0
test_images = test_images / 255.0



index = np.random.randint(train_images.shape[0])
plt.figure()
plt.imshow(train_images[index])
plt.grid(False)
plt.title('Image #{} : '.format(index) + class_names[train_labels[index]])
plt.show()

# breakpoint()




model_file_name = 'model.h5'
from tensorflow.keras.models import load_model
model = tf.keras.Sequential([
    #First layer
    tf.keras.layers.Conv2D(128,kernel_size=(3,3), activation = 'linear', input_shape = (150, 150, 3)), # the nn will learn the good filter to use
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='same'),
    tf.keras.layers.Dropout(0.25),

    #Second layer
    tf.keras.layers.Conv2D(256, (3, 3)),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),
    tf.keras.layers.Dropout(0.25),

    #Third layer
    tf.keras.layers.Conv2D(256, (3, 3)),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='same'),
    tf.keras.layers.Dropout(0.4),


    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='linear'),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(nb_classes, activation='softmax')

])

model.summary()
SVG(model_to_dot(model).create(prog='dot', format='svg'))
Utils.plot_model(model,to_file='model.png',show_shapes=True)

model.compile(optimizer=tf.keras.optimizers.Adam(), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
keras_callbacks = [
    ModelCheckpoint(model_file_name,
                    monitor='acc',
                    save_best_only=True,
                    verbose=1),
    EarlyStopping(monitor='acc', patience=2, verbose=0)
]

history = model.fit(train_images, train_labels, batch_size=64, epochs=10, validation_split = 0.2, verbose=1, callbacks=keras_callbacks)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

print('pllllllooooootttttttssss!!!!!!')

# test_loss = model.evaluate(test_images, test_labels)


model = load_model(model_file_name)


index = np.random.randint(test_images.shape[0]) # We choose a random index

img = (np.expand_dims(test_images[index], 0))
predictions = model.predict(img)     # Vector of probabilities
pred_img = np.argmax(predictions[0]) # We take the highest probability
pred_label = class_names[pred_img]
true_label = class_names[test_labels[index]]

title = 'Pred : {} and True : {}  '.format(pred_label,true_label)

plt.figure()
plt.imshow(test_images[index])
plt.grid(False)
plt.title(title)
plt.show()


test_loss = model.evaluate(test_images, test_labels)
print('Model Accuracy: ',round(int(test_loss[1]*100)),'%')


