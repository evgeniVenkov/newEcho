import pdfplumber

# Открываем PDF-файл
with pdfplumber.open("C:\\Users\\admin\\Pictures\\book.pdf") as pdf:
    text = ""

    # Чтение текста с каждой страницы
    for page in pdf.pages:
        text += page.extract_text()


with open("./data/his_rus.txt", "w",encoding = "utf-8") as f:
    f.write(text)
print(text)
