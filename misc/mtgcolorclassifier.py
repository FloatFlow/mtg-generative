import os
import argparse
import numpy as np
from keras.layers import BatchNormalization, Dense
from keras.layers import Activation, Input, Flatten
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description='Parameters for miniGAN')

    # general parameters
    parser.add_argument('--train_x',
                        type=str,
                        default='Dataset/color_palettes/mtg_card_palettes_top8.npy',
                        help='Extracted dominant color palettes')
    parser.add_argument('--train_y',
                        type=str,
                        default='Dataset/color_palettes/mtg_card_labels.npy',
                        help='MTG colors for each card')

    return parser.parse_args()

class MtgColorClassifier():
    def __init__(self,
                 x,
                 y):
        self.x = np.load(x)
        self.y = np.load(y)

    def build_model(self):
        optimizer = SGD(lr=1e-4, momentum=0.9)

        model_in = Input(shape=(8, 3))
        x = Flatten()(model_in)

        x = Dense(8*3)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Dense(5)(x)
        model_out = Activation('sigmoid')(x)

        self.model = Model(model_in, model_out)
        self.model.compile(optimizer,
                           'binary_crossentropy',
                           metrics=['accuracy'])

    def train(self):
        self.x = (self.x/127.5)-1.0
        #self.y = np.expand_dims(self.y, axis=-1)
        self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(self.x,
                                                                              self.y,
                                                                              test_size=0.2,
                                                                              random_state=42)

        checkpointing = ModelCheckpoint('training/model_saves/mtg_color_classifier.h5',
                                        monitor='val_loss',
                                        verbose=1,
                                        save_best_only=True)


        history = self.model.fit(x=self.train_x,
                                 y=self.train_y,
                                 batch_size=32,
                                 epochs=1000,
                                 verbose=2,
                                 callbacks=[checkpointing],
                                 validation_data=(self.val_x,
                                                  self.val_y)
                                 )


def main():
    args = parse_args()

    colormodel = MtgColorClassifier(x=args.train_x,
                                    y=args.train_y)
    colormodel.build_model()
    colormodel.train()    

if __name__ == '__main__':
    main()