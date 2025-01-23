import requests
from bs4 import BeautifulSoup
import json
helper = False
# Указываем URL сайта, откуда будем парсить данные
URL = "https://djvu.online/file/fUiLgJ8TWC6fV"  # Замените на нужный сайт

# 1. Отправляем GET-запрос к сайту
response = requests.get(URL)

# Проверяем успешность запроса
if response.status_code == 200:
    # 2. Загружаем содержимое страницы
    soup = BeautifulSoup(response.text, "html.parser")

    # 3. Ищем нужные элементы (например, заголовки и текст)
    # Предположим, что заголовки находятся в теге <h2>, а текст в <p>
    text = soup.find(id="dump-body")
    print(text)
    with open('../data/history_book1.txt',"w",encoding = "utf-8")as f:
        f.write(text.get_text())



    print("Данные успешно сохранены в history_textbook.json!")
else:
    print(f"Ошибка: не удалось получить данные. Код состояния {response.status_code}")

# Основные моменты:
# - Убедитесь, что URL доступен и данные, которые вы ищете, находятся на сайте.
# - Используйте инструменты браузера (DevTools) для анализа структуры HTML.
# - Не забывайте соблюдать правила парсинга и пользоваться сайтами в рамках закона.

if helper:
    print(soup.find("h1"))  # По тегу
    print(soup.find(id="main"))  # По ID
    print(soup.find_all(class_="text"))  # По классу
    print(soup.select("div p.text"))  # По CSS-селектору