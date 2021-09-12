import os
import argparse
import numpy as np
import pandas as pd
from keras.layers import BatchNormalization, Dense, Conv1D
from keras.layers import Activation, Input, Flatten, LeakyReLU
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import colorgram
from multiprocessing import Process, Queue, Lock, Value
import time
from functools import partial

def parse_args():
    parser = argparse.ArgumentParser(description='Parameters for small CNN')

    # general parameters
    parser.add_argument('--train_x',
                        type=str,
                        default='data/color_palettes/mtg_card_palettes_top8.npy',
                        help='Extracted dominant color palettes')
    parser.add_argument('--train_y',
                        type=str,
                        default='data/color_palettes/mtg_card_labels.npy',
                        help='MTG colors for each card')
    parser.add_argument('--epochs',
                        type=int,
                        default=0)

    parser.add_argument('--load_weights',
                        type=bool,
                        default=True)
    parser.add_argument('--model_weights',
                        type=str,
                        default='logging/model_saves/mtg_palette_classifier/mtg_color_classifier_8colors.h5')
    parser.add_argument('--labeling_dir',
                        type=str,
                        default='data/imaginaryreddit_images')
    parser.add_argument('--n_cpu',
                        type=int,
                        default=4)

    return parser.parse_args()

def onehot_label_decoder(one_hot_label):
    # convert labels to one-hot encoded
    str_label = []
    for i, label in enumerate(one_hot_label):
        if (i == 0) and (label == 1):
            str_label.append('W')
        if (i == 1) and (label == 1):
            str_label.append('B')
        if (i == 2) and (label == 1):
            str_label.append('U')
        if (i == 3) and (label == 1):
            str_label.append('G')
        if (i == 4) and (label == 1):
            str_label.append('R')
    return str(str_label)

class PaletteGenerator():
    def __init__(self,
                 img_dir,
                 batch_size,
                 n_cpu,
                 n_palette,
                 file_type=('.jpg', '.png', '.jpeg')):
        self.img_dir = img_dir
        self.file_type = file_type
        self.batch_size = batch_size
        self.n_cpu = n_cpu
        self.n_palette = n_palette
        self.queue = Queue(maxsize=self.n_cpu*4)
        self.lock = Lock()
        self.run = True
        self.counter = Value('i', -1)
        self.generate_data_paths()

        for _ in range(self.n_cpu):
             p = Process(target=self.fetch_batch)
             p.start()

    def generate_data_paths(self):
        img_paths = []
        for file in os.listdir(self.img_dir):
            if file.endswith(self.file_type):
                img_paths.append(os.path.join(file))

        # store paths and labels
        self.df = pd.DataFrame({'paths':img_paths})
        self.df.drop_duplicates(inplace=True)
        if self.df.shape[0] % self.batch_size != 0:
            self.n_batches = (self.df.shape[0]//self.batch_size) + 1
        else:
            self.n_batches = (self.df.shape[0]//self.batch_size)
        self.indices = np.arange(self.df.shape[0])

    def fetch_batch(self):
        while self.run:

            while self.queue.full():
                time.sleep(0.1)

            self.lock.acquire()
            self.counter.value += 1
            if self.counter.value >= self.n_batches:
                self.counter.value = 0
            end_position = min((self.counter.value+1)*self.batch_size, len(self.indices))
            positions = self.indices[self.counter.value*self.batch_size:end_position]
            self.lock.release()
            file_batch = self.df['paths'].iloc[positions].values
            
            palettes = []
            for img_path in file_batch:
                palette_objs = colorgram.extract(os.path.join(self.img_dir, img_path), 
                                                 self.n_palette)
                dom_palette = [[color.rgb.r, color.rgb.g, color.rgb.b] for color in palette_objs]
                n_results = len(dom_palette)
                while len(dom_palette) < self.n_palette:
                    idx = np.random.randint(n_results)
                    dom_palette.append(dom_palette[idx])
                dom_palette = np.array(dom_palette)
                palettes.append(dom_palette)
            
            palettes = np.array(palettes)

            self.queue.put((palettes, file_batch))

    def next(self):
        while self.queue.empty():
            time.sleep(0.1)
        return self.queue.get()

    def shuffle(self):
        self.df = self.df.sample(frac=1.0).reset_index(drop=True)

    def end(self):
        self.run = False
        self.queue.close()

class MtgColorClassifier():
    def __init__(self,
                 x,
                 y):
        self.x = np.load(x)
        self.y = np.load(y)
        self.n_palette = int(x.split('top')[-1].split('.')[0])

    def build_model(self, ch=64):
        #optimizer = SGD(lr=1e-4, momentum=0.9)
        optimizer = Adam(lr=1e-4)

        model_in = Input(shape=(self.n_palette, 3))

        x = Conv1D(filters=ch,
                   kernel_size=1,
                   padding='valid')(model_in)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Flatten()(x)
        x = Dense(5)(x)
        model_out = Activation('sigmoid')(x)

        self.model = Model(model_in, model_out)
        self.model.compile(optimizer,
                           'binary_crossentropy',
                           metrics=['accuracy'])
        print(self.model.summary())

    def train(self, epochs=100, batch_size=32):
        self.x = (self.x/127.5)-1.0
        #self.y = np.expand_dims(self.y, axis=-1)
        self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(self.x,
                                                                              self.y,
                                                                              test_size=0.2,
                                                                              random_state=42)

        model_save = '../logging/model_saves/mtg_palette_classifier/mtg_color_classifier_{}colors.h5'.format(self.n_palette)
        checkpointing = ModelCheckpoint(model_save,
                                        monitor='val_loss',
                                        verbose=1,
                                        save_best_only=True)


        history = self.model.fit(x=self.train_x,
                                 y=self.train_y,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=2,
                                 callbacks=[checkpointing],
                                 validation_data=(self.val_x,
                                                  self.val_y)
                                 )

    def iter_label(self, labeling_dir):
        img_paths = []

        for root, dirnames, filenames in os.walk(labeling_dir):
            for file in filenames:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    img_paths.append(os.path.join(root, file))
        img_paths = list(set(img_paths))

        pbar = tqdm(total=len(img_paths))
        palettes = []
        for img_path in img_paths:
            palette_objs = colorgram.extract(img_path, self.n_palette)
            dom_palette = [[color.rgb.r, color.rgb.g, color.rgb.b] for color in palette_objs]
            n_results = len(dom_palette)
            while len(dom_palette) < self.n_palette:
                idx = np.random.randint(n_results)
                dom_palette.append(dom_palette[idx])
            dom_palette = np.array(dom_palette)
            
            y_pred = self.model.predict(np.expand_dims(dom_palette, axis=0))
            y_pred = np.where(y_pred > 0.5, 1, 0)
            y_pred = [onehot_label_decoder(onehot_label) for onehot_label in y_pred]
            new_fname = '{}_{}.png'.format(os.path.basename(img_path).split('.')[0], str(y_pred[0]))
            try:
                os.rename(img_path,
                          os.path.join(os.path.dirname(img_path), new_fname))
            except FileExistsError:
                print('{} Already Renamed'.format(new_fname))
                continue
            pbar.update()
        pbar.close()

    def label(self, batch_size, labeling_dir, n_cpu):
        n_batches = (len(list(os.listdir(labeling_dir)))//batch_size) + 1
        palette_generator = PaletteGenerator(labeling_dir,
                                             n_cpu=n_cpu,
                                             batch_size=batch_size,
                                             n_palette=self.n_palette)

        img_paths = []
        predicted_labels = []
        pbar = tqdm(total=n_batches)
        for batch_n in range(n_batches):
            palettes, img_names = palette_generator.next()

            # predict class
            y_pred = self.model.predict(palettes)
            y_pred = np.where(y_pred > 0.5, 1, 0)
            y_pred = [onehot_label_decoder(onehot_label) for onehot_label in y_pred]

            img_paths.append(img_names)
            predicted_labels.append(y_pred)
            pbar.update()
        pbar.close()

        img_paths =  [item for sublist in img_paths for item in sublist]
        predicted_labels = [item for sublist in predicted_labels for item in sublist]
        rename_df = pd.DataFrame({'filename':img_paths, 'label':predicted_labels})
        rename_df.drop_duplicates(inplace=True)
        rename_df['new_fname'] = ['{}_{}.png'.format(os.path.basename(fname).split('.')[0], label) \
                                  for fname, label in zip(rename_df['filename'], rename_df['label'])]
        rename_df.to_csv('mtg_palette_rename.csv')
        pbar = tqdm(total=rename_df.shape[0])
        for i, row in rename_df.iterrows():
            os.rename(os.path.join(labeling_dir, row['filename']),
                      os.path.join(labeling_dir, row['new_fname']))
            pbar.update()
        pbar.close()
        

def main():
    args = parse_args()

    colormodel = MtgColorClassifier(x=args.train_x,
                                    y=args.train_y)
    colormodel.build_model()

    if args.load_weights:
        colormodel.model.load_weights(args.model_weights)

    if args.epochs > 0:
        colormodel.train(epochs=args.epochs)    

    #colormodel.label(batch_size=32,
    #                 labeling_dir=args.labeling_dir,
    #                 n_cpu=args.n_cpu)
    colormodel.iter_label(labeling_dir=args.labeling_dir)

if __name__ == '__main__':
    main()