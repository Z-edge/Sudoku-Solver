from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import activations
from sklearn.metrics import classification_report
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler


# Function to define the layers of the CNN model and create it
#############################################################################
def createModel(height, width, channels, num_labels):
    shape = (height, width, channels)

    layer_1 = Conv2D(32,
                     kernel_size=3,
                     input_shape=shape,
                     activation=activations.relu)
    layer_2 = BatchNormalization()
    layer_3 = Conv2D(32,
                     kernel_size=3,
                     activation=activations.relu)
    layer_4 = BatchNormalization()
    layer_5 = Conv2D(32,
                     kernel_size=5,
                     strides=2,
                     padding='same',
                     activation=activations.relu)
    layer_6 = BatchNormalization()
    layer_7 = Dropout(0.4)
    layer_8 = Conv2D(64,
                     kernel_size=3,
                     input_shape=shape,
                     activation=activations.relu)
    layer_9 = BatchNormalization()
    layer_10 = Conv2D(64,
                      kernel_size=3,
                      activation=activations.relu)
    layer_11 = BatchNormalization()
    layer_12 = Conv2D(64,
                      kernel_size=5,
                      strides=2,
                      padding='same',
                      activation=activations.relu)
    layer_13 = BatchNormalization()
    layer_14 = Dropout(0.4)
    layer_15 = Flatten()
    layer_16 = Dense(128, activation=activations.relu)
    layer_17 = BatchNormalization()
    layer_18 = Dropout(0.4)
    layer_19 = Dense(num_labels, activation=activations.softmax)

    model = Sequential(
        [
         layer_1,
         layer_2,
         layer_3,
         layer_4,
         layer_5,
         layer_6,
         layer_7,
         layer_8,
         layer_9,
         layer_10,
         layer_11,
         layer_12,
         layer_13,
         layer_14,
         layer_15,
         layer_16,
         layer_17,
         layer_18,
         layer_19
        ]
    )
    return model
#############################################################################


# Function to compile the model
#############################################################################
def compileModel(model):
    print('[Compiling model.....]')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
#############################################################################


# Function to train the model on the MNIST training Dataset
#############################################################################
def trainModel(model, train_X, train_y, test_X, test_y):
    print('[Training model.....]')
    callback = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** (x+25))
    history = model.fit(train_X, train_y,
                        validation_data=(test_X, test_y),
                        batch_size=64,
                        epochs=25,
                        steps_per_epoch=train_X.shape[0]//64,
                        callbacks=[callback],
                        verbose=1)
    print(history)
    return model
#############################################################################


# Function to evaluate the model on the MNIST testing dataset
#############################################################################
def evaluateModel(model, test_X, test_y):
    print('[Evaluating model.....]')
    pred_y = model.predict(test_X,
                           verbose=1)

    print(classification_report(test_y.argmax(axis=1),
                                pred_y.argmax(axis=1),
                                target_names=['0', '1', '2', '3', '4',
                                              '5', '6', '7', '8', '9']))
    return model
#############################################################################


# Loading the MNIST dataset, reshaping the datasets
# and calling above functionalities
#############################################################################
((train_X, train_y), (test_X, test_y)) = mnist.load_data()

train_X = train_X.reshape(list(train_X.shape)+[1])
test_X = test_X.reshape(list(test_X.shape)+[1])

train_X = train_X.astype('float32') / 255
test_X = test_X.astype('float32') / 255

train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

num_labels = test_y.shape[1]

model = createModel(28, 28, 1, num_labels)
model = compileModel(model)
model = trainModel(model, train_X, train_y, test_X, test_y)
model = evaluateModel(model, test_X, test_y)
#############################################################################


# Saving the trained model
#############################################################################
model_name = 'My_Model1'
model.save('Models/' + model_name, save_format='h5')
print("Model saved successfully!")
#############################################################################
