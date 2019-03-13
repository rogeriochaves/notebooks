import unittest

from scrappers.funda_nl import find_houses
from tests.helpers import load_fixture
from unittest.mock import Mock


def mocked_browser(fixture_file, url):
    browser = Mock()
    browser.get = Mock()
    browser.page_source = load_fixture("funda_nl", fixture_file, url)
    return browser


class FundaNlComTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.maxDiff = None

    def test_find_data_from_search_results_page(self):
        browser = mocked_browser(
            "koop_amsterdam.html", "https://www.funda.nl/en/koop/amsterdam/")
        result = find_houses(browser)

        self.assertEqual(
            result[0], {
                'address': 'Oterleekstraat 15 1023 ED Amsterdam',
                'price': '€ 319,000 k.k.',
                'living_area': '63',
                'plot_size': '54',
                'rooms': '4',
                'url': 'https://www.funda.nl/en/koop/amsterdam/huis-40009279-oterleekstraat-15/'
            })

        self.assertEqual(
            result[1], {
                'address': 'Bastenakenstraat 142 1066 JG Amsterdam',
                'price': '€ 850,000 k.k.',
                'living_area': '151',
                'plot_size': '200',
                'rooms': '4',
                'url': 'https://www.funda.nl/en/koop/amsterdam/huis-86457819-bastenakenstraat-142/'
            })

    def test_find_data_from_search_results_second_page(self):
        browser = mocked_browser(
            "koop_amsterdam_p2.html", "https://www.funda.nl/en/koop/amsterdam/p2/")
        result = find_houses(browser, 2)

        self.assertEqual(
            result[0], {
                'address': 'Haya van Someren-Downerstraat 32 1067 WR Amsterdam',
                'price': '€ 450,000 k.k.',
                'living_area': '129',
                'plot_size': '204',
                'rooms': '5',
                'url': 'https://www.funda.nl/en/koop/amsterdam/huis-40843916-haya-van-someren-downerstraat-32/'
            })

    def test_resilient_against_missing_data(self):
        browser = mocked_browser(
            "koop_amsterdam_p104.html", "https://www.funda.nl/en/koop/amsterdam/p104/")
        result = find_houses(browser, 104)

        self.assertEqual(
            result[0], {
                'address': 'Silodam 1013 AW Amsterdam',
                'price': '€ 22,500 k.k.',
                'living_area': None,
                'plot_size': None,
                'rooms': None,
                'url': 'https://www.funda.nl/en/koop/amsterdam/parkeergelegenheid-40005099-silodam/'
            })
