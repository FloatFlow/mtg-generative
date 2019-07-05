import os
import cv2
import pandas as pd
import numpy as np
import colorgram
from PIL import Image
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Parameters for miniGAN')

    # general parameters
    parser.add_argument('--img_dir',
                        default='Dataset/images',
                        help='Directory of images')
    parser.add_argument('--n_colors',
                        type=int,
                        default=8,
                        help='')


    return parser.parse_args()

def label_parser(label_name):
    lookup_dict = {'W':[1,0,0,0,0], 'B':[0,1,0,0,0], 'U':[0,0,1,0,0], 'G':[0,0,0,1,0], 'R':[0,0,0,0,1]}

    temp = [0,0,0,0,0]
    for k, v in lookup_dict.items():
        if k in label_name:
            temp = [sum(x) for x in zip(temp, v)]

    return temp


def main():
    args = parse_args()


    img_paths = list(os.listdir(args.img_dir))
    palettes = []
    labels = []

    pbar = tqdm(total=len(img_paths))
    for img_file in img_paths:
        label = os.path.basename(img_file).split('.')[0].split('_')[-1]
        onehot_label = label_parser(label)
        #img = Image.open(img_file)
        palette_objs = colorgram.extract(os.path.join(args.img_dir, img_file), 
                                         args.n_colors)
        dom_palette = np.array([color.rgb for color in palette_objs])
        palettes.append(dom_palette)
        labels.append(onehot_label)
        pbar.update()
    pbar.close()

    palettes = np.array(palettes)
    labels = np.array(labels)
    print('Palettes shape: {}'.format(palettes.shape))
    print('Labels shape: {}'.format(labels.shape))

    palette_path = os.path.join('Dataset',
                                'color_palettes',
                                'mtg_card_palettes_top{}.npy'.format(args.n_colors))
    np.save(palette_path,
            palettes)

    label_path = os.path.join('Dataset',
                              'color_palettes',
                              'mtg_card_labels.npy')
    np.save(label_path, 
            labels)
    print('Job Finished!')


if __name__ == '__main__':
    main()