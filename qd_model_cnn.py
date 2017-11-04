#!/usr/bin/env python

"""
This class builds a CNN model on the qd dataset
@author : Hardeep Arora
@date   : 05-Oct-2017
"""

from keras.layers import Conv2D, MaxPooling2D, Dropout, Input, Concatenate, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta, Adam
from keras.models import Model
from quick_draw_dataset import QuickDraw
from matplotlib import pyplot as plt

class QDModel:

    _model = None
    _nb_classes = None
    _img_cols = None
    _img_rows = None
    _img_channels = None
    _x_train = None
    _x_test  = None
    _y_train = None
    _y_test  = None
    _input_shape = None
    _batch_size = 64
    _epochs = 50
    _hist = None

    def __init__(self):
        qd = QuickDraw()
        self._x_train, self._x_test, self._y_train, self._y_test = qd.load_data(500000, 30000)
        self._nb_classes = qd.get_classes()
        self._img_rows, self._img_cols, self._img_channels = qd.get_img_dim()
        self._input_shape = (self._img_rows, self._img_cols, self._img_channels)

    @staticmethod
    def _set_checkpoint():
        filepath="checkpoints/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        return callbacks_list

    def build_model(self):
        Inp = Input(shape=self._input_shape, name='Input_01')

        conv1 = Conv2D(32, kernel_size=(5,5), activation='relu', name='Conv_01',padding='same')(Inp)
        conv2 = Conv2D(32, kernel_size=(3,3), activation='relu', name='Conv_02',padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2,2), name='MaxPool_01')(conv2)
        drop1 = Dropout(0.25, name='Dropout_01')(pool1)

        conv11 = Conv2D(32, kernel_size=(5,5), activation='relu', name='Conv_011',padding='same')(drop1)
        conv21 = Conv2D(32, kernel_size=(3,3), activation='relu', name='Conv_021',padding='same')(conv11)
        pool11 = MaxPooling2D(pool_size=(2,2), name='MaxPool_011')(conv21)
        drop11 = Dropout(0.25, name='Dropout_011')(pool11)

        flat1 = Flatten(name='Flatten_01')(drop11)
        dense1= Dense(1024, activation='relu', name='Dense_01')(flat1)
        drop2 = Dropout(0.5, name='Dropout_02')(dense1)
        dense2= Dense(512, activation='relu', name='Dense_02')(drop2)
        Output= Dense(self._nb_classes, activation='softmax', name='Output')(dense2)

        self._model = Model(Inp, Output)

    def train(self):

        self._model.summary()
        self._model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

        self._hist = self._model.fit(self._x_train, self._y_train,
                                    batch_size=self._batch_size,
                                    epochs=self._epochs,
                                    verbose=1,
                                    callbacks = QDModel._set_checkpoint(),
                                    validation_data=(self._x_test, self._y_test))

    def evaluate(self):
        # Evaluate model with test data set and share sample prediction results
        evaluation = self._model.evaluate(self._x_test, self._y_test,
                                        batch_size=self._batch_size)
        print('Model Accuracy = %.2f' % (evaluation[1]))
        print('Model Loss = %.2f' % (evaluation[0]))

    def plot_train(self):
        h = self._hist.history
        if 'acc' in h:
            meas='acc'
            loc='lower right'
        else:
            meas='loss'
            loc='upper right'
        plt.plot(self._hist.history[meas])
        plt.plot(self._hist.history['val_'+meas])
        plt.title('model '+meas)
        plt.ylabel(meas)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc=loc)
        plt.show()

    def save(self,filename):
        """
            This method saves the model definition and weights as *.h5
        :return: nothing
        """
        # serialize model to JSON
        model_json = self._model.to_json()
        with open('model/' + filename + ".json", "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        self._model.save_weights('model/' + filename + ".h5")
        print("Saved model to disk")


def main():
    model = QDModel()
    model.build_model()
    model.train()
    model.evaluate()
    #model.plot_train()
    model.save("qd_200_10k_adam")

if __name__ == "__main__":
    main()



