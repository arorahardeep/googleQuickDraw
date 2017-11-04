#!/usr/bin/env python

"""
This class loads the google quick draw dataset
@author : Hardeep Arora
@date   : 05-Oct-2017
"""

import numpy as np
import keras



class QuickDraw:

    img_rows = 28
    img_cols = 28
    img_channels = 1

    _num_classes = None
    _num_examples_per_class = 10000
    _classes = ['aircraft carrier','airplane','alarm clock','ambulance','angel','ant',
                'anvil','apple','axe','banana','bandage','barn','baseball bat',
                'baseball','basket','basketball','bathtub','beach','bear','beard',
                'bed','bee','belt','bicycle','binoculars','birthday cake',
                'blueberry','book','boomerang','bottlecap','bowtie','bracelet',
                'brain','bread','broom','bulldozer','bus','bush','butterfly',
                'cactus','cake','calculator','calendar','camel','camera','campfire',
                'candle','cannon','canoe','car','carrot','cat','cello','chandelier',
                'clock','cloud','coffee cup','compass','computer','cookie','couch',
                'cow','crab','crayon','crocodile','crown','cup','diamond','dog',
                'dolphin','donut','dragon','dresser','drill','drums','duck',
                'dumbbell','ear','elbow','elephant','envelope','eraser',
                'eye','eyeglasses','face','fan','feather','fence','finger','fire hydrant',
                'fireplace','firetruck','fish','flamingo','flashlight','flip flops',
                'floor lamp','flower','flying saucer','foot','fork','frog','frying pan',
                'garden hose','garden','giraffe','goatee','golf club','grapes','grass',
                'guitar','hamburger','hammer','hand','harp','hat','headphones',
                'hedgehog','helicopter','helmet','hexagon','hockey puck',
                'hockey stick','horse','hospital','hot air balloon','hot dog',
                'hot tub','hourglass','house plant','house','hurricane',
                'ice cream','jacket','jail','kangaroo','key','keyboard','knee',
                'knife','ladder','lantern','laptop','leaf','leg','light bulb',
                'lighter','lighthouse','lightning','line','lion','lipstick','lobster',
                'lollipop','mailbox','map','marker','matches','megaphone',
                'mermaid','microphone','microwave','monkey',
                'moon','mosquito','motorbike','mountain','mouse','moustache',
                'mouth','mug','mushroom','nail','necklace','nose','ocean',
                'octagon','octopus','onion','oven','owl',
                'paint can','paintbrush','palm tree','panda','pants',
                'paper clip','parachute','parrot','passport','peanut',
                'pear','peas','pencil','penguin','piano','pickup truck',
                'picture frame','pig','pillow']

    def __init__(self):
        self._num_classes = len(self._classes)

    def _unison_shuffled_copies(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]



    def load_data(self, n_train, n_test):
        """
        This function loads the dataset
        :return:
        """
        x_data = np.load("data/x_data_200_classes_10k.npy")
        labels = [np.full((self._num_examples_per_class,), self._classes.index(qdraw)) for qdraw in self._classes]
        y_data = np.concatenate(labels,axis=0)

        x_data,y_data = self._unison_shuffled_copies(x_data,y_data)

        x_train = x_data[n_test:n_train+n_test]
        x_test  = x_data[-n_test:]
        y_train = y_data[n_test:n_train+n_test]
        y_test  = y_data[-n_test:]

        # One-Hot encode y
        y_train = keras.utils.to_categorical(y_train)
        y_test  = keras.utils.to_categorical(y_test)

        # Normalize x
        x_train = x_train/255
        x_test  = x_test/255

        # Reshape
        x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, self.img_channels)
        x_test  = x_test.reshape (x_test.shape[0] , self.img_rows, self.img_cols, self.img_channels)

        print("x_train = " + str(x_train.shape))
        print("x_test = " + str(x_test.shape))
        print("y_train = " + str(y_train.shape))
        print("y_test = " + str(y_test.shape))

        return x_train, x_test, y_train, y_test

    def get_classes(self):
        return self._num_classes

    def get_img_dim(self):
        return self.img_rows, self.img_cols, self.img_channels

def main():
    qd = QuickDraw()
    qd.load_data(10000,1000)

if __name__ == "__main__":
    main()





