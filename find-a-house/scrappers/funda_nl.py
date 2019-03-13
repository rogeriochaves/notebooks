from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import re
import time


def get_url(browser, url):
    browser.get(url)
    time.sleep(1.5)
    while (re.findall('We denken dat je een robot bent', browser.page_source) or
           re.findall('Unable To Identify Your Browser', browser.page_source)):
        print('Bot detected, please solve captcha')
        time.sleep(10)
    return BeautifulSoup(browser.page_source, 'html.parser')


def find_houses(browser, page=1):
    page_num = f'p{page}/' if page > 1 else ''
    print('Scrapping page', page)
    soup = get_url(
        browser, f'https://www.funda.nl/en/koop/amsterdam/{page_num}')
    contents = soup.select('.search-result-content')
    if (len(contents) <= 0):
        raise Exception('no houses info')

    return [extract_house_data(content) for content in contents]


def extract_house_data(content):
    url = content.select('.search-result-header > a')[0]['href']
    address = re.sub(r'\s{2,}', ' ', content.select('h3')[0].text.strip())
    price = content.select('.search-result-price')[0].text.strip()

    living_area, plot_size, rooms = (None, None, None)
    details = content.select('.search-result-kenmerken li')
    if (len(details) > 0):
        sizes = details[0].select('span')
        living_area = sizes[0].text.strip().split(' ')[0]
        plot_size = sizes[1].text.strip().split(
            ' ')[0] if len(sizes) > 1 else None
    if (len(details) > 1):
        rooms = details[1].text.strip().split(' ')[0]

    return {
        'address': address,
        'price': price,
        'living_area': living_area,
        'plot_size': plot_size,
        'rooms': rooms,
        'url': f'https://www.funda.nl{url.replace("https://www.funda.nl", "")}',
    }


def start_browser(url):
    browser = webdriver.Firefox(
        executable_path='bin/geckodriver')
    browser.set_page_load_timeout(30)
    browser.get(url)
    time.sleep(5)

    return browser


def save_or_append_csv(data):
    if (len(data) > 0):
        df = pd.DataFrame(data)
        try:
            df_curr = pd.read_csv('funda.csv')
            print("Found existing csv, appending data")
            df = pd.concat([df_curr, df])
        except Exception:
            pass
        df.to_csv("funda.csv", index=False)


if __name__ == "__main__":
    initial_page = 1
    final_page = 185
    print('Start scrapping funda from page', initial_page, 'to', final_page)

    browser = start_browser('https://www.funda.nl/mijn/login/')

    all_houses = []
    try:
        for page in range(initial_page, final_page + 1):
            all_houses += find_houses(browser, page)
    except Exception as e:
        print(e)
        print(f'Stopped at page {page}')
    print('> Found', len(all_houses), 'total')

    save_or_append_csv(all_houses)
    browser.close()
