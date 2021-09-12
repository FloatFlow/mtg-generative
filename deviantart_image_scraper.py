from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import os
import json
from urllib.request import urlopen, Request
import sys
import time
import argparse
import requests
from tqdm import tqdm
import traceback
from bs4 import BeautifulSoup

# adding path to geckodriver to the OS environment variable
# assuming that it is stored at the same path as this script
#os.environ["PATH"] += os.pathsep + os.getcwd()
def parse_args():
    parser = argparse.ArgumentParser(description='Parameters for image scrape')
    # general parameters
    parser.add_argument(
        '--output_dir',
        type=str,
        default="D:/Projects/mtg-minigan/data/anime_girls"
        )
    parser.add_argument(
        '--search_terms',
        type=list,
        default=['anime girl', 'girl', 'woman']#'mahou shoujo']
        )
    parser.add_argument(
        '--n_images',
        type=int,
        default=10000
        )
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    for term in args.search_terms:
        print('Current Search Term: {}'.format(term))
        main_searcher(term, args.n_images, args.output_dir)
        time.sleep(30)

def main_searcher(searchtext, n_images, output_dir):
    url = 'https://www.deviantart.com/search/deviations?q={}'.format(
        searchtext.replace(' ', '+'),
        )
    driver = webdriver.Firefox()
    driver.get(url)

    headers = {}
    headers['User-Agent'] = "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36"
    extensions = {"jpg", "jpeg", "png", "gif"}    

    image_url_list = []
    for _ in range(n_images//20):
        try:
            driver.execute_script("window.scrollBy(0, 1000000)")
            new_images = driver.find_elements_by_tag_name('img')
            for new_img in new_images:
                img_url = new_img.get_attribute('src')
                if img_url is None:
                    img_url = new_img.get_attribute('data-src')
                if img_url is None:
                    img_url = new_img.get_attribute('href')

                if (img_url is not None) and (img_url not in image_url_list):
                    image_url_list.append(img_url)
                if img_url is None:
                    html = new_img.get_attribute('outerHTML')
                    attrs = BeautifulSoup(html, 'html.parser').a.attrs
                    print(attrs)
            time.sleep(1.5)
        except Exception:
            continue
    
    downloaded_img_count = 0
    pbar = tqdm(total=len(image_url_list))
    for img_url in image_url_list:
        try:
            sess = requests.Session()
            req = sess.get(img_url, headers={'User-Agent': 'Firefox'})
            f = open(
                os.path.join(
                    output_dir,
                    f"{str(searchtext.replace(' ', '_'))}_{str(downloaded_img_count)}.png"
                    ),
                "wb"
                )
            f.write(req.content)
            f.close()
            downloaded_img_count += 1
        except Exception:
            traceback.print_exc()
        pbar.update()
    pbar.close()

    print("Total downloaded: ", downloaded_img_count, "/", len(image_url_list))
    driver.quit()

if __name__ == '__main__':
    main()