import unittest
import re
import os.path
import requests

def trim(text):
    return re.sub('\s|\n|\t', '', text)


def read_file(filename):
    try:
        fh = open(filename, "r")
        data = fh.read()
        fh.close()

        return data
    except:
        return None


def load_fixture(scrapper_name, test_name, url):
    filename = os.path.join(os.path.dirname(__file__),
                            "fixtures", scrapper_name, test_name)
    data = read_file(filename)
    if data:
        return data

    data = requests.get(url).text

    fp = open(filename, "w")
    fp.write(data)
    fp.close()
    return data
