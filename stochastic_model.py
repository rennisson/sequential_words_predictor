from collections import Counter
from functools import wraps
from pathlib import Path
import argparse
import glob
import heapq
import pickle
import time

CURRENT_WORD = 0
NEXT_WORD = 1

def time_measurement(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[{func.__name__}] Execution time: {end - start:.4f}s")
        return result
    return wrapper 

@time_measurement
def get_text():
    contents = []

    for file in glob.glob("books/*.txt"):
        with open(file, 'r', encoding='utf-8') as f:
            # Lê todo o conteúdo e armazena na variável 'texto'
            contents.append(f.read())
    text = "\n".join(contents)
    return text

@time_measurement
def format_text(text):
    text_cleaned = text.replace(",", " ")
    text_cleaned = text_cleaned.replace(".", " ")
    text_cleaned = text_cleaned.replace(";", " ")
    text_cleaned = text_cleaned.replace(":", " ")
    text_cleaned = text_cleaned.replace("?", " ")
    text_cleaned = text_cleaned.replace("...", " ")
    text_cleaned = text_cleaned.replace("!", " ")
    text_cleaned = text_cleaned.replace("--", " ")
    text_cleaned = text_cleaned.replace("\n", " ")
    text_cleaned = text_cleaned.replace('"', " ")
    text_cleaned = text_cleaned.lower()
    text_cleaned = text_cleaned.split()

    return text_cleaned

@time_measurement
def get_words_map(text):
    words_map = {}

    for i in range(len(text) - 1):
        current_word = text[i]
        next_word    = text[i+1]

        if (current_word, next_word) in words_map:
            words_map[(current_word, next_word)] = words_map[(current_word, next_word)] + 1
        elif (current_word, next_word) not in words_map:
            words_map[(current_word, next_word)] = 1

    return words_map

@time_measurement
def marginal_probabilities(text):
    N = len(text)
    print(f"{N=}")
    words_counter = Counter(text)
    marginal_probs = {key: freq / N for key, freq in words_counter.items()}

    return marginal_probs

@time_measurement
def conditional_probabilities(words_map, N, alpha=1):
    # Calculating CONDITIONAL PROBABILITIES
    words_frequencies = {}
    conditional_probs = {}

    for word_combination, frequency in words_map.items():
        # Sum all occurencies of the 'word' in the 'next_word' position
        if word_combination[1] in words_frequencies:
            words_frequencies[word_combination[1]] += frequency
        else:
            words_frequencies[word_combination[1]] = frequency
        
    for words, freq in words_map.items():
        conditional_probs[words] = (freq + alpha) / (words_frequencies[words[1]]  + (alpha*N))
    
    return conditional_probs, words_frequencies


def total_probability(current_word, words_frequencies, conditional_probs, marginal_probs, N, alpha=1):
    # CALCULO DA PROBABILIDADE TOTAL
    total_probability = 0
    for word in marginal_probs.keys():
        conditional_prob = conditional_probs[(current_word, word)] \
                            if   (current_word, word) in conditional_probs \
                            else (alpha / (words_frequencies.get(word, 0) + (alpha * N)))
        
        priori = marginal_probs[word]
        total_probability += conditional_prob * priori

    return total_probability


def find_next_candidates(current_word, conditional_probs, marginal_probs, words_frequencies, N, alpha=1):
    next_candidates_probs = {}
    next_candidates = marginal_probs.keys() 
    total_prob = total_probability(current_word, words_frequencies, conditional_probs, marginal_probs, N, alpha)

    if total_prob <= 0:
        return {'.': 0}

    for candidate in next_candidates:
        conditional_prob = conditional_probs[(current_word, candidate)] \
                            if (current_word, candidate) in conditional_probs \
                            else (alpha / (words_frequencies.get(candidate, 0) + (alpha * N)))
        
        next_candidates_probs[(current_word, candidate)] = (conditional_prob * marginal_probs[candidate]) / total_prob
    
    return next_candidates_probs

@time_measurement
def train():
    print("Getting texts...")
    text = get_text()

    print("Cleaning text...")
    clean_text = format_text(text)

    print("Starting model training...")
    print("Mapping words...")
    words_map = get_words_map(clean_text)

    print("Calculating marginal probabilities...")
    marginal_probs = marginal_probabilities(clean_text)
    N = len(marginal_probs.keys())

    print("Calculating conditional probabilities...")
    conditional_probs, words_frequencies = conditional_probabilities(words_map, N, alpha=1)
    
    print("Training finished.")

    with open("cond_probs.pkl", "wb") as f:
        pickle.dump(conditional_probs, f)
    
    with open("words_frequencies.pkl", "wb") as f:
        pickle.dump(words_frequencies, f)
    
    with open("marginal_probs.pkl", "wb") as f:
        pickle.dump(marginal_probs, f)
    
    with open("len_vocabulary.pkl", "wb") as f:
        pickle.dump(N, f)
        
    return conditional_probs, words_frequencies, marginal_probs, N


def main(phrase: str, length: int):

    conditional_probs = {}
    words_frequencies = {}
    marginal_probs = {}
    N = 0

    if Path('cond_probs.pkl').exists():
        with open(Path('cond_probs.pkl'), "rb") as f:
            conditional_probs = pickle.load(f)

        with open(Path("words_frequencies.pkl"), "rb") as f:
            words_frequencies = pickle.load(f)
        
        with open(Path("marginal_probs.pkl"), "rb") as f:
           marginal_probs = pickle.load(f)
        
        with open(Path("len_vocabulary.pkl"), "rb") as f:
            N = pickle.load(f)

        print("Dados recuperados com sucesso!")
    else:
        conditional_probs, words_frequencies, marginal_probs, N = train()

    print("Performing predictions...")
    phrase = phrase.lower()
    for _ in range(length):
        current_word = phrase.split()[-1]
        print(f"{current_word=}")

        next_candidates_probs = find_next_candidates(current_word, conditional_probs, marginal_probs, words_frequencies, N, alpha=1)

        best_candidate = max(next_candidates_probs, key=next_candidates_probs.get)
        phrase = phrase + " " + best_candidate[1]
    print(f"Final phrase: {phrase}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preditor de palavras")
    parser.add_argument("--phrase", type=str, required=True, help="A frase inicial para predição")
    parser.add_argument("--length", type=int, default=1, help="A quantidade de palavras que quero prever")
    
    args = parser.parse_args()
    main(args.phrase, args.length)