from .time_measurement import time_measurement
from typing import List
import glob

@time_measurement
def get_text() -> List:
    """
    Localiza, lê e consolida todos os arquivos de texto do diretório de livros.

    A função utiliza o módulo glob para iterar por todos os arquivos '.txt' na 
    pasta 'books/', extrai seu conteúdo bruto e os une em uma única string gigante. 
    Após a leitura, delega a limpeza dos dados para a função format_text.

    Returns:
        list: Uma lista de strings contendo todas as palavras (tokens) de todos 
              os livros, já normalizadas e limpas.
    """
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
def format_text(text: str) -> List:
    """
    Realiza a limpeza, normalização e tokenização do texto bruto.

    Esta função remove pontuações, quebras de linha e caracteres especiais, 
    substituindo-os por espaços para evitar a junção indevida de palavras. 
    O texto é convertido para minúsculas (case folding) e dividido em uma 
    lista de palavras individuais.

    Args:
        text (str): A string bruta contendo todo o conteúdo dos arquivos lidos.

    Returns:
        list: Uma lista de palavras (tokens) limpas, sem pontuação e em 
              letras minúsculas.
    """
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