from .stochastic_model import total_probability
import heapq
import random

STOP_WORDS   = [
    'o', 'a', 'os', 'as', 'de', 'do', 'da', 'dos', 'das', 
    'em', 'no', 'na', 'nos', 'nas', 'um', 'uma', 'uns', 'umas',
    'e', 'que', 'com', 'por', 'para', 'se', 'meu', 'seu', 'não'
]

def find_next_candidates(current_word, conditional_probs, marginal_probs, words_frequencies, N, alpha=1):
    next_candidates_probs = {}
    next_candidates = marginal_probs.keys() 
    total_prob = total_probability(current_word, words_frequencies, conditional_probs, marginal_probs, N, alpha)

    for candidate in next_candidates:
        conditional_prob = conditional_probs[(current_word, candidate)] \
                            if   (current_word, candidate) in conditional_probs \
                            else (alpha / (words_frequencies.get(candidate, 0) + (alpha * N)))
        
        next_candidates_probs[(current_word, candidate)] = (conditional_prob * marginal_probs[candidate]) / total_prob
    
    return next_candidates_probs


def choose_next_word(next_candidates_probs):
    top_30_candidates = heapq.nlargest(
                                30, 
                                next_candidates_probs.items(), 
                                key=lambda item: item[1]
                            )

    candidates = []
    weights = []

    if random.choice([0,1]) == 0:
        candidates = [item[0] for item in top_30_candidates if item[0][1] not in STOP_WORDS]
        weights    = [item[1] for item in top_30_candidates if item[0][1] not in STOP_WORDS]
    else:
        candidates = [item[0] for item in top_30_candidates]
        weights    = [item[1] for item in top_30_candidates]

    best_candidate = random.choices(candidates, weights=weights, k=1)[0]
    _, choosen_word = best_candidate
    return choosen_word