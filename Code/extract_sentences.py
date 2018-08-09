import sys
import os
from bs4 import BeautifulSoup
import re


def main():
	dir_name = sys.argv[1]
	sentences_file = open(sys.argv[2], "w")
	for _file in os.listdir(dir_name):
		html_file = open(dir_name + "/" + str(_file), "r")
		soup = BeautifulSoup(html_file, 'html.parser') 
		all_text = soup.find_all('p')
		for raw_text in all_text:
			text = re.sub('\n', "", raw_text.text.strip())
			sentences_file.write("%s\n" % (text))  
		html_file.close()


if __name__ == "__main__":
	main()


