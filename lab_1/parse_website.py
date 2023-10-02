from bs4 import BeautifulSoup
import requests
import json

url = "https://services.dtkt.ua/catalogues/indexes/3"

page = requests.get(url)

soup = BeautifulSoup(page.text, 'html.parser')

table = soup.find('table', class_='in_text_tab table table-bordered table-services')

year_row = table.find_all('tr')[0]
year_values = [int(cell.text.strip()) for cell in year_row.find_all('td')[1:-1] if cell.text.strip()]

value_row = table.find_all('tr')[-1]
value_values = [float(cell.text.replace(',', '.').strip()) for cell in value_row.find_all('td')[1:] if
                cell.text.strip()]

data = {
    "year": year_values,
    "value": value_values
}

with open('data.json', 'w') as f1:
    json.dump(data, f1)
