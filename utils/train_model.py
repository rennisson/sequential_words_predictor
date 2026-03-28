from .save_model import save_model
from .stochastic_model import (
    conditional_probabilities,
    get_words_map,
    marginal_probabilities
)
from .text import get_text
from .time_measurement import time_measurement 

@time_measurement
def train():
    text = get_text()

    print("Starting model training...")
    words_map      = get_words_map(text)
    marginal_probs = marginal_probabilities(text)
    N              = len(marginal_probs.keys())
    conditional_probs, words_frequencies = conditional_probabilities(
                                                words_map, 
                                                N, 
                                                alpha=1
                                            )
    
    print("Training finished.")

    save_model(marginal_probs, N, conditional_probs, words_frequencies)
        
    return conditional_probs, words_frequencies, marginal_probs, N