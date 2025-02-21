import os

# Путь к папке с файлами
folder_path = "C:/Users/admin/PycharmProjects/newEcho/data/"

# Проходим по всем файлам в папке
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)

    # Проверяем, является ли это текстовым файлом
    if os.path.isfile(file_path) and file_name.endswith(".txt"):
        try:
            # Пытаемся открыть файл с кодировкой utf-8
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read()  # Читаем файл, чтобы проверить его на наличие ошибок
        except UnicodeDecodeError:
            # Если возникла ошибка, файл не соответствует кодировке UTF-8
            print(f"Файл {file_name} не соответствует UTF-8")
