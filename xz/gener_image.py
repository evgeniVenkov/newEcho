from PIL import Image, ImageDraw, ImageFont
import os
import random

# Папка для сохранения изображений
output_dir = "./equal_sign_images"
os.makedirs(output_dir, exist_ok=True)

# Параметры генерации
image_size = (256, 128)  # Размер изображения
num_images = 1000  # Количество изображений
font_size_range = (40, 100)  # Диапазон размера шрифта
font_path = "C:\\Windows\\Fonts\\arial.ttf"  # Путь к шрифту

if not os.path.exists(font_path):
    raise FileNotFoundError(f"Шрифт не найден по пути: {font_path}")

# Генерация изображений
def get_text():




for i in range(num_images):
    # Создаём пустое изображение
    image = Image.new("RGB", image_size, color="white")
    draw = ImageDraw.Draw(image)

    # Выбираем случайный размер шрифта и позицию
    font_size = random.randint(*font_size_range)
    font = ImageFont.truetype(font_path, font_size)

    # Рисуем знак "=" в случайной позиции
    text = get_text()
    text_width, text_height = font.getbbox(text)[2:4]


    x = random.randint(0, image_size[0] - text_width)
    y = random.randint(0, image_size[1] - text_height)
    draw.text((x, y), text, fill="black", font=font)

    # Сохраняем изображение
    image.save(os.path.join(output_dir, f"equal_{i}.png"))

print(f"Сгенерировано {num_images} изображений в папке '{output_dir}'")