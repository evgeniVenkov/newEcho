import requests
from bs4 import BeautifulSoup
import json


def fetch_wikipedia_article(title: str):
    """
    Загружает статью с Википедии по названию.
    :param title: Название статьи на Википедии (без "Википедия:" и других префиксов).
    :return: Текст статьи.
    """
    url = f"https://ru.wikipedia.org/wiki/{title}"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Ошибка загрузки страницы: {url}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')

    # Извлекаем только текст статьи
    content_div = soup.find('div', {'class': 'mw-parser-output'})
    paragraphs = content_div.find_all('p')
    # Собираем текст из параграфов
    article_text = ""
    for p in paragraphs:
        article_text += p.get_text()

    # Очищаем текст от лишнего
    article_text = article_text.replace("\n", " ").strip()

    return article_text


def save_article_to_json(title: str, text: str):
    """
    Сохраняет статью в формате JSON.
    :param title: Название статьи.
    :param text: Текст статьи.
    """
    new_data = {'title': title, 'text': text}

    with open(f"./data/my_data.json", "r+", encoding="utf-8") as f:
        # Прочитаем существующие данные
        try:
            f.seek(0)
            existing_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            existing_data = []

        # Добавим новые данные
        existing_data.append(new_data)

        # Запишем обновленные данные в файл
        f.seek(0)
        json.dump(existing_data, f, ensure_ascii=False, indent=4)


# Пример использования
# article_title = 'Компьютерная_безопасность'  # Название статьи без "Википедия:"
# article_text = fetch_wikipedia_article(article_title)

# if article_text:
#     save_article_to_json(article_title, article_text)
#     print("good :-D")

from datasets import load_dataset

dataset = load_dataset("SiberiaSoft/SiberianPersonaChat")


# # Сохранение текстов в текстовый файл
with open("siberian_persona_chat.txt", "w", encoding="utf-8") as f:
    for dialogue in dataset["train"]:
        # Каждый диалог можно разделить на вопросы и ответы
        f.write(f"Question: {dialogue['input']}\n")

