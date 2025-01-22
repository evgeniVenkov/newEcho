from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Путь для кэша модели
project_dir = "./model_cache"

# Загрузка токенизатора и модели
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=project_dir)
model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=project_dir)

# Текст на русском для проверки
text = "Искусственный интеллект это"
input_ids = tokenizer.encode(text, return_tensors="pt")

# Генерация текста
output = model.generate(input_ids, max_length=50, temperature=0.7)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Сгенерированный текст:")
print(generated_text)
