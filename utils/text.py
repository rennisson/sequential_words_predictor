from .time_measurement import time_measurement
import glob

@time_measurement
def get_text():
    print("Getting texts...")
    contents = []

    for file in glob.glob("books/*.txt"):
        with open(file, 'r', encoding='utf-8') as f:
            # Lê todo o conteúdo e armazena na variável 'texto'
            contents.append(f.read())
    text = "\n".join(contents)

    clean_text = format_text(text)
    return clean_text

@time_measurement
def format_text(text):
    print("Cleaning text...")
    text_cleaned = text.replace(",", " ")
    text_cleaned = text_cleaned.replace(".", " ")
    text_cleaned = text_cleaned.replace(";", " ")
    text_cleaned = text_cleaned.replace(":", " ")
    text_cleaned = text_cleaned.replace("?", " ")
    text_cleaned = text_cleaned.replace("...", " ")
    text_cleaned = text_cleaned.replace("!", " ")
    text_cleaned = text_cleaned.replace("--", " ")
    text_cleaned = text_cleaned.replace("—", " ")
    text_cleaned = text_cleaned.replace("\n", " ")
    text_cleaned = text_cleaned.replace('"', " ")
    text_cleaned = text_cleaned.lower()
    text_cleaned = text_cleaned.split()

    return text_cleaned