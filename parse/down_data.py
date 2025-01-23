from datasets import load_dataset

dataset = load_dataset("SiberiaSoft/SiberianPersonaChat")

# Сохранение текстов в текстовый файл
with open("../data/siberian_persona_chat.txt", "w", encoding="utf-8") as f:
    for dialogue in dataset["train"]:
        # Каждый диалог можно разделить на вопросы и ответы

        f.write(f"Question: {dialogue['input']}\n")
        f.write(f"Answer: {dialogue['output']}\n")
        f.write("\n")
