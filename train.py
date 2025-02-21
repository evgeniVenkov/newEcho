from transformers import GPT2LMHeadModel, GPT2Tokenizer,AdamW
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import os
import torch


# Проверяем, доступен ли GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

print(f"Training will be done on: {device}")

# Загружаем модель и токенизатор
model_path = "C:/Users/admin/PycharmProjects/newEcho/results"
book_folder_path = "C:/Users/admin/PycharmProjects/newEcho/data"


# Загружаем все книги из папки
book_paths = [os.path.join(book_folder_path, book) for book in os.listdir(book_folder_path) if book.endswith(".txt")]

# Загружаем несколько файлов в датасет
dataset = load_dataset("text", data_files={"train": book_paths}, split="train")


model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Устанавливаем eos_token как токен для паддинга
tokenizer.pad_token = tokenizer.eos_token

model = model.to(device)
# Функция для токенизации

# optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
def tokenize_function(examples):
    return tokenizer(examples["text"], return_tensors="pt", padding=True, truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Параметры для тренировки
training_args = TrainingArguments(
    output_dir="./results",  # Директория для сохранения результатов тренировки
    num_train_epochs=1,  # Количество эпох (проходов по данным)
    per_device_train_batch_size=4,  # Размер батча (количество примеров на одном шаге)
    save_steps=10_000,  # Частота сохранения модели (каждые 10,000 шагов)
    save_total_limit=2,  # Максимальное количество сохраненных моделей

)

# Переопределяем класс Trainer для добавления вычисления потерь
class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Эта функция переопределяет стандартное вычисление потерь.
        Она добавляет 'labels' в inputs и вычисляет потери на основе 'input_ids'.
        """
        inputs = {key: value.to(device) for key, value in inputs.items()}
        inputs['labels'] = inputs['input_ids']  # Устанавливаем labels равными input_ids для вычисления потерь
        outputs = model(**inputs)  # Прогоняем модель через входные данные
        loss = outputs.loss  # Получаем значение потерь
        # Если необходимо, возвращаем как потери, так и выходные данные модели, иначе только потери
        return (loss, outputs) if return_outputs else loss

# Создаем объект тренера с указанными параметрами
trainer = MyTrainer(
    model=model,  # Модель, которую будем обучать
    args=training_args,  # Параметры тренировки
    data_collator=None,  # Метод для группировки данных в батчи (по умолчанию None)
    train_dataset=tokenized_datasets,  # Обучающий датасет с токенизированными данными
)

# Тренировка
trainer.train()

# Сохраняем модель один раз в конце тренировки
model.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)
print("Saved model checkpoint to {}".format(training_args.output_dir))