# -*- coding: utf-8 -*-
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Указываем путь к модели и токенизатору на локальном диске
model_path = "./results"

# Загружаем модель и токенизатор с локального пути
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Устанавливаем pad_token_id, если его нет
tokenizer.pad_token = tokenizer.eos_token

# Функция для генерации ответа на вопрос
def ask_question(question):
    # Токенизируем вопрос, передаем attention_mask
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    # Генерируем ответ
    outputs = model.generate(inputs['input_ids'], max_length=100, attention_mask=inputs['attention_mask'])
    # Декодируем ответ
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Пример использования
question = "Как дела?"
answer = ask_question(question)
print(answer)
