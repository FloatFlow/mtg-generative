from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import os
import json
from urllib.request import urlopen, Request
import sys
import time
import argparse
from tqdm import tqdm
import traceback

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
        default=['anime girl', 'mahou shoujo']
        )
    parser.add_argument(
        '--start_year',
        type=int,
        default=2010
        )
    parser.add_argument(
        '--end_year',
        type=int,
        default=2021
        )
    parser.add_argument(
        '--n_per_search_year',
        type=int,
        default=2000
        )
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    for year in range(args.start_year, args.end_year+1):
        for term in args.search_terms:
            print('\n\n ############################################## \n\n')
            print('Year of Images: {}'.format(year))
            print('Current Search Term: {}'.format(term))
            print('\n\n ############################################## \n')
            main_searcher(term, year, args.n_per_search_year, args.output_dir)
            time.sleep(30)

def main_searcher(searchtext, searchyear, n_per_search_year, output_dir):
    num_requested = n_per_search_year # number of images to download
    number_of_scrolls = int(num_requested / 400 + 1 )
    # number_of_scrolls * 400 images will be opened in the browser

    #if not os.path.exists(download_path + searchtext.replace(" ", "_")):
    #    os.makedirs(download_path + searchtext.replace(" ", "_"))

    #url = "https://www.google.co.in/search?q="+searchtext+"&source=lnms&tbm=isch" # for general, non-time restricted search
    # color only, photo only, time restricted search (1 year spans):
    #url = 'https://www.google.com/search?q={}&client=firefox-b-1-ab&biw=1536&bih=727&source=lnt&tbs=ic%3Acolor%2Citp%3Aphoto%2Ccdr%3A1%2Ccd_min%3A1%2F1%2F{}%2Ccd_max%3A{}&tbm=isch'.format(searchtext, searchyear, searchyear)
    url = 'https://www.google.com/search?q={}&tbs=cdr:1,cd_min:{},cd_max:{}&source=lnms&tbm=isch&sa=X&ved=2ahUKEwi_54PWx_PyAhVMGFkFHfEVBR0Q_AUoAXoECAEQAw&biw=1283&bih=670'.format(
        searchtext.replace(' ', '+'),
        str(searchyear),
        str(searchyear+1)
        )
    driver = webdriver.Firefox()
    driver.get(url)

    headers = {}
    headers['User-Agent'] = "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36"
    extensions = {"jpg", "jpeg", "png", "gif"}
    img_count = 0
    downloaded_img_count = 0

    """
    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        time.sleep(0.5)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")

        if new_height == last_height:
        #break #insert press load more
            try:
                driver.find_element_by_xpath("//input[@value='Show more results']").click()
            except:
                break
        last_height = new_height
    """
    while True:
        for __ in range(10):
            # multiple scrolls needed to show all 400 images
            driver.execute_script("window.scrollBy(0, 1000000)")
            time.sleep(0.2)
        # to load next 400 images
        time.sleep(0.5)
        try:
            driver.find_element_by_xpath("//input[@value='Show more results']").click()
        except Exception:
            break
    

    # imges = driver.find_elements_by_xpath('//div[@class="rg_meta"]') # not working anymore
    #imges = driver.find_elements_by_xpath('//div[contains(@class,"rg_meta")]')
    imges = driver.find_elements_by_xpath('//img[contains(@class,"rg_i")]')
    pbar = tqdm(total=len(imges))
    for img in imges:
        img_count += 1
        img_url = img.get_attribute('src')
        if img_url is None:
            img_url = img.get_attribute('data-src')
        try:
            req = Request(img_url, headers=headers)
            raw_img = urlopen(req).read()
            f = open(
                os.path.join(
                    output_dir,
                    f"{str(searchtext.replace(' ', '_'))}_{str(searchyear)}_{str(downloaded_img_count)}.png"
                    ),
                "wb"
                )
            f.write(raw_img)
            f.close()
            downloaded_img_count += 1
        except Exception:
            traceback.print_exc()
        pbar.update()
    pbar.close()

    print("Total downloaded: ", downloaded_img_count, "/", img_count)
    driver.quit()

if __name__ == '__main__':
    main()