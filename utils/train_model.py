from .save_model import save_model
from .stochastic_model import (
    conditional_probabilities,
    words_frequencies_map,
    marginal_probabilities
)
from .text import get_text
from .time_measurement import time_measurement 
from typing import Dict, Tuple

@time_measurement
def train() -> Tuple[Dict, Dict, Dict, int]:
    """
    Orquestra o pipeline completo de treinamento do modelo probabilístico.

    O processo consiste em:
    1. Carregar o texto bruto do dataset (Machado de Assis).
    2. Gerar o mapa de frequências de bigramas (coocorrência).
    3. Calcular as probabilidades marginais (a priori) de cada palavra.
    4. Calcular as probabilidades condicionais P(B|A) utilizando Laplace Smoothing.
    5. Persistir todos os artefatos calculados em arquivos para uso futuro.

    Returns:
        tuple: Uma tupla contendo os quatro pilares do modelo:
            - conditional_probs (dict): Probabilidades de transição entre palavras.
            - words_frequencies (dict): Frequências acumuladas de sucessores.
            - marginal_probs (dict): Probabilidades individuais de cada palavra.
            - N (int): O tamanho do vocabulário (número de palavras únicas).
    """
    text = get_text()

    print("Starting model training...")
    words_map      = words_frequencies_map(text)
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