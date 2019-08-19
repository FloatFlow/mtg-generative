import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from keras.layers import BatchNormalization, Dense, Conv2D
from keras.layers import Activation, Input, Flatten, LeakyReLU, MaxPooling2D
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model.utils import *
from model.network_blocks import *

def parse_args():
    parser = argparse.ArgumentParser(description='Parameters for small CNN')

    # general parameters
    parser.add_argument('--train_dir',
                        type=str,
                        default='data/mtg_images')
    parser.add_argument('--img_dim',
                        type=int,
                        default=256)
    parser.add_argument('--batch_size',
                        type=int,
                        default=16)
    parser.add_argument('--epochs',
                        type=int,
                        default=100)

    parser.add_argument('--n_cpu',
                        type=int,
                        default=3)
    parser.add_argument('--load_weights',
                        type=bool,
                        default=False)
    parser.add_argument('--model_weights',
                        type=str,
                        default='logging/model_saves/mtg_image_classifier/mtg_image_classifier.h5')
    parser.add_argument('--labeling_dir',
                        type=str,
                        default='data/test_labeling')

    return parser.parse_args()



class MtgImageClassifier():
    def __init__(self,
                 img_dim):
        self.img_dim = img_dim
        self.build_model()

    def build_model(self, ch=64):
        optimizer = SGD(lr=1e-4, momentum=0.9)
        #optimizer = Adam(lr=1e-4)

        model_in = Input(shape=(self.img_dim, self.img_dim, 3))

        x = Conv2D(filters=ch,
                   kernel_size=7,
                   strides=2,
                   padding='same')(model_in)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x) #64x64

        ch *= 2
        x = res_block(x, ch, downsample=True) #32x32x64
        ch *= 2
        x = res_block(x, ch, downsample=True) #16x16x128
        ch *= 2
        x = res_block(x, ch, downsample=True) #8x8x256

        #x = Flatten()(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(5)(x)
        model_out = Activation('sigmoid')(x)

        self.model = Model(model_in, model_out)
        self.model.compile(optimizer,
                           'binary_crossentropy',
                           metrics=['accuracy'])
        print(self.model.summary())

    def train(self, training_dir, n_cpu, epochs=100, batch_size=32):
        img_paths = []
        img_labels = []
        for file in os.listdir(training_dir):
            if file.endswith(('.jpg', '.png', '.jpeg', '.mp4')):
                img_paths.append(os.path.join(training_dir, file))
                label = os.path.basename(file).split('.')[0].split('_')[-1]
                label = onehot_label_encoder(label)
                img_labels.append(label)

        train_x, val_x, train_y, val_y = train_test_split(img_paths,
                                                          img_labels,
                                                          test_size=0.2,
                                                          random_state=42)

        train_generator = KerasImageGenerator(x=train_x,
                                              y=train_y,
                                              batch_size=batch_size,
                                              img_dim=self.img_dim)

        val_generator = KerasImageGenerator(x=val_x,
                                            y=val_y,
                                            batch_size=batch_size,
                                            img_dim=self.img_dim)

        model_save = 'logging/model_saves/mtg_image_classifier/mtg_image_classifier.h5'
        checkpointing = ModelCheckpoint(model_save,
                                        monitor='val_loss',
                                        verbose=1,
                                        save_best_only=True)


        history = self.model.fit_generator(generator=train_generator,
                                           steps_per_epoch=len(train_x)//batch_size,
                                           epochs=epochs,
                                           verbose=1,
                                           validation_data=val_generator,
                                           validation_steps=len(val_x)//batch_size,
                                           use_multiprocessing=True,
                                           workers=n_cpu,
                                           max_queue_size=n_cpu*4,
                                           shuffle=True)

    def label(self, batch_size, labeling_dir):
        path_df = pd.DataFrame({'filename': list(os.listdir(labeling_dir))})
        predicted_labels = []

        n_batches = (path_df.shape[0]//batch_size) + 1

        pbar = tqdm(total=n_batches)
        for batch_n in range(n_batches):
            # define batch
            if batch_n != (n_batches - 1):
                file_batch = path_df['filename'].iloc[batch_n*batch_size:(batch_n+1)*batch_size]
            else:
                file_batch = path_df['filename'].iloc[batch_n*batch_size:]

            # preprocessing; extract palettes
            palettes = []
            for img_file in file_batch:
                palette_objs = colorgram.extract(os.path.join(labeling_dir, img_file), 
                                                 self.n_palette)
                dom_palette = [color.rgb for color in palette_objs]
                n_results = len(dom_palette)
                while len(dom_palette) < self.n_palette:
                    idx = np.random.randint(n_results)
                    dom_palette.append(dom_palette[idx])
                dom_palette = np.array(dom_palette)
                palettes.append(dom_palette)
            
            palettes = np.array(palettes)

            # predict class
            y_pred = self.model.predict(palettes)
            y_pred = np.where(y_pred > 0.5, 1, 0)
            y_pred = [onehot_label_decoder(onehot_label) for onehot_label in y_pred]
            predicted_labels.append(y_pred)

            pbar.update()

        pbar.close()
        path_df['label'] = predicted_labels[0]
        path_df['new_filename'] = ['{}_{}.png'.format(fname.split('.')[0], label) for fname, label in zip(path_df['filename'], path_df['label'])]

        for i, row in path_df.iterrows():
            os.rename(os.path.join(labeling_dir, row['filename']),
                      os.path.join(labeling_dir, row['new_filename']))

def main():
    args = parse_args()

    classifier = MtgImageClassifier(img_dim=args.img_dim)

    if args.load_weights:
        classifier.model.load_weights(args.model_weights)
        print('Model Weights Loaded')

    if args.epochs > 0:
        classifier.train(training_dir=args.train_dir,
                         epochs=args.epochs,
                         n_cpu=args.n_cpu,
                         batch_size=args.batch_size)    

    classifier.label(batch_size=32, labeling_dir=args.labeling_dir)

if __name__ == '__main__':
    main()