from pathlib import Path
from utils.predict import (
    choose_next_word,
    find_next_candidates
)
from utils.save_model import get_file
from utils.train_model import train
import argparse


def main(phrase: str, length: int) -> None:
    """
    Ponto de entrada principal que gerencia o ciclo de vida da predição de texto.

    A função executa as seguintes etapas:
    1. Verifica a existência de arquivos de modelo pré-treinados (cache) na pasta 'pkl_files'.
    2. Caso os arquivos não existam, dispara o processo de treinamento completo.
    3. Caso existam, carrega os dados serializados para a memória.
    4. Executa um loop de predição iterativo: a cada iteração, identifica a última 
       palavra da frase atual, prediz a sucessora mais provável (considerando o 
       componente aleatório e filtro de stop words) e a concatena à frase original.

    Args:
        phrase (str): A frase ou palavra inicial fornecida pelo usuário para 
            começar a geração de texto.
        length (int): A quantidade de palavras adicionais que o modelo deve 
            gerar após a frase inicial.

    Returns:
        None: A função imprime a frase final consolidada diretamente no console.
    """

    conditional_probs = {}
    words_frequencies = {}
    marginal_probs = {}
    N = 0

    path = Path(__file__).parent / "pkl_files"
    conditional_probs_path = path / "cond_probs.pkl"
    words_frequencies_path = path / "words_frequencies.pkl"
    marginal_probs_path = path / "marginal_probs.pkl"
    len_vocabulary_path = path / "len_vocabulary.pkl"
    
    if not conditional_probs_path.exists() \
        or not words_frequencies_path.exists() \
        or not marginal_probs_path.exists() \
        or not len_vocabulary_path.exists():

        conditional_probs, words_frequencies, marginal_probs, N = train()

    else:
        conditional_probs = get_file(conditional_probs_path)
        words_frequencies = get_file(words_frequencies_path)
        marginal_probs = get_file(marginal_probs_path)
        N = get_file(len_vocabulary_path)
        print("Model obtained!")
        

    print("Performing predictions...")
    phrase = phrase.lower()
    for _ in range(length):
        current_word = phrase.split()[-1]

        next_candidates_probs = find_next_candidates(
                                    current_word, 
                                    conditional_probs, 
                                    marginal_probs, 
                                    words_frequencies, 
                                    N, 
                                    alpha=1
                                )
        
        choosen_word = choose_next_word(next_candidates_probs)
        phrase = phrase + " " + choosen_word
        
    print(f"FINAL PHRASE: {phrase}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preditor de palavras")
    parser.add_argument("--phrase", type=str, help="Initial phrase to prediction")
    parser.add_argument("--length", type=int, default=1, help="Quantity of words to predict")
    parser.add_argument("--train", type=bool, default=False, help="Flag for model training")
    
    args = parser.parse_args()

    if args.train:
        train()
    else:
        main(args.phrase, args.length)