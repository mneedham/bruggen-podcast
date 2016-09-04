from bs4 import BeautifulSoup, Tag
from soupselect import select

import requests
import os

page = BeautifulSoup(open("bruggen.htm", "r"), "html.parser")

for item in select(page, "ul.posts li"):
    link = select(item, "a")[0]
    if "podcast" in link.text.lower():
        href = link.get("href")
        print href
        
        file_name = "data/{0}".format(href.split("/")[-1])
        if not os.path.isfile(file_name):
            page = requests.get(href)
            with open(file_name, 'wb') as test:
                test.write(page.content)
