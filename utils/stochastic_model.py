from collections import Counter
from .time_measurement import time_measurement

@time_measurement
def get_words_map(text):
    print("Mapping words...")
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
    print("Calculating marginal probabilities...")

    N = len(text)
    print(f"{N=}")
    words_counter = Counter(text)
    marginal_probs = {key: freq / N for key, freq in words_counter.items()}

    return marginal_probs

@time_measurement
def conditional_probabilities(words_map, N, alpha=1):
    print("Calculating conditional probabilities...")
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