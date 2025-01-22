from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

# Указываем путь к уже скачанной модели
model_dir = "../gpt2_medium"

# 1. Загружаем токенизатор и модель
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)

# 2. Загружаем текстовый датасет
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")  # Можно заменить на свой датасет

# 3. Токенизация данных
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 4. Настраиваем параметры обучения
training_args = TrainingArguments(
    output_dir="./gpt2_xl_finetuned",  # Папка для сохранения модели
    evaluation_strategy="epoch",  # Оценка после каждой эпохи
    learning_rate=5e-5,  # Начальная скорость обучения
    num_train_epochs=3,  # Количество эпох
    per_device_train_batch_size=2,  # Размер батча
    save_steps=500,  # Сохранение каждые 500 шагов
    save_total_limit=2,  # Хранить только 2 последних чекпоинта
    logging_dir="./logs",  # Логи обучения
    logging_steps=10,  # Логи каждые 10 шагов
    fp16=True,  # Использование 16-битной точности
)

# 5. Настраиваем Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# 6. Запускаем обучение
trainer.train()

# 7. Сохраняем обученную модель
model.save_pretrained("./gpt_medium_little_tuned")
tokenizer.save_pretrained("./gpt_medium_little_tuned ")
