from PIL import Image
import glob, os
from tqdm import tqdm
import argparse

IMG_DIR = 'agglomerated_images'

def main():
    img_paths = [os.path.join(IMG_DIR, fpath) for fpath in os.listdir(IMG_DIR)]

    pbar = tqdm(total=len(img_paths))
    for fpath in img_paths:
        try:
            im = Image.open(fpath)
            im.save(fpath)
        except OSError:
            print('Deleting {}'.format(fpath))
            os.remove(fpath)
            continue
        pbar.update()
    pbar.close()

    print("Success: Image Sanitization Complete")
if __name__ == '__main__':
    main()