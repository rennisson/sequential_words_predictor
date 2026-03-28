from collections import Counter
from functools import wraps
from pathlib import Path
import argparse
import glob
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
def conditional_probabilities(text, words_map, position_key=NEXT_WORD):
    # Calculating CONDITIONAL PROBABILITIES
    words_frequencies = {}
    conditional_probs = {}
    for word in text:
        if word not in words_frequencies:
            # Sum all occurencies of the 'word' in the 'next_word' position
            word_frequency = {key: freq for key, freq in words_map.items() if key[position_key] == word}
            words_frequencies[word] = sum(word_frequency.values())
        
            for key, freq in word_frequency.items():
                conditional_probs[key] = freq / words_frequencies[word] if words_frequencies[word] > 0 else 0
    
    return conditional_probs, words_frequencies


def total_probability(current_word, words_frequencies, conditional_probs, marginal_probs):
    # CALCULO DA PROBABILIDADE TOTAL
    total_probability = 0
    for word in words_frequencies.keys():
        # inverse_conditional_probs, _ = conditional_probabilities(clean_text, position_key=CURRENT_WORD)
        conditional_prob = conditional_probs[(current_word, word)] if (current_word, word) in conditional_probs else 0
        priori = marginal_probs[word]
        total_probability += conditional_prob * priori

    return total_probability


def find_next_candidates(current_word, conditional_probs, marginal_probs, words_map, words_frequencies):
    next_candidates_probs = {}
    next_candidates = [key for key in words_map.keys() if key[CURRENT_WORD] == current_word]
    total_prob = total_probability(current_word, words_frequencies, conditional_probs, marginal_probs)

    if total_prob <= 0:
        return {'.': 0}

    for candidate in next_candidates:
        # print(f"P{candidate}*P({candidate[1]})")
        # print(f"{conditional_probs.get(candidate, 0)=}")
        # print(f"{marginal_probs[candidate[1]]=}")
        conditional_prob = conditional_probs[candidate] if candidate in conditional_probs else 0
        next_candidates_probs[candidate] = (conditional_prob * marginal_probs[candidate[1]]) / total_prob
    
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

    print("Calculating conditional probabilities...")
    conditional_probs, words_frequencies = conditional_probabilities(clean_text, words_map, position_key=NEXT_WORD)
    print("Calculating marginal probabilities...")
    marginal_probs = marginal_probabilities(clean_text)
    print("Training finished.")

    with open("cond_probs.pkl", "wb") as f:
        pickle.dump(conditional_probs, f)
    
    with open("words_frequencies.pkl", "wb") as f:
        pickle.dump(words_frequencies, f)
    
    with open("marginal_probs.pkl", "wb") as f:
        pickle.dump(marginal_probs, f)
    
    with open("words_map.pkl", "wb") as f:
        pickle.dump(words_map, f)
        
    return words_map,conditional_probs,words_frequencies,marginal_probs


def main(phrase: str, length: int):

    words_map = {}
    conditional_probs = {}
    words_frequencies = {}
    marginal_probs = {}

    if Path('cond_probs.pkl').exists():
        with open(Path('cond_probs.pkl'), "rb") as f:
            conditional_probs = pickle.load(f)

        with open(Path("words_frequencies.pkl"), "rb") as f:
            words_frequencies = pickle.load(f)
        
        with open(Path("marginal_probs.pkl"), "rb") as f:
           marginal_probs = pickle.load(f)
        
        with open(Path("words_map.pkl"), "rb") as f:
            words_map = pickle.load(f)

        print("Dados recuperados com sucesso!")
    else:
        words_map, conditional_probs, words_frequencies, marginal_probs = train()

    print("Performing predictions...")
    phrase = phrase.lower()
    for _ in range(length):
        current_word = phrase.split()[-1]
        print(f"{current_word=}")

        next_candidates_probs = find_next_candidates(current_word, conditional_probs, marginal_probs, words_map, words_frequencies)

        # print(f"{words_frequencies=}")
        # print(f"{marginal_probs=}")
        # print(f"{conditional_probs=}")
        # print(f"{next_candidates_probs=}")

        best_candidate = max(next_candidates_probs, key=next_candidates_probs.get)
        # print(f"Best candidate: '{best_candidate[1]}', with probability {next_candidates_probs[best_candidate]}")
        phrase = phrase + " " + best_candidate[1]
    print(f"Final phrase: {phrase}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preditor de palavras")
    parser.add_argument("--phrase", type=str, required=True, help="A frase inicial para predição")
    parser.add_argument("--length", type=int, default=1, help="A quantidade de palavras que quero prever")
    
    args = parser.parse_args()
    main(args.phrase, args.length)