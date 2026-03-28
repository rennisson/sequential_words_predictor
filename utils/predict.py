from .stochastic_model import total_probability
from typing import Dict, Optional
import heapq
import random

STOP_WORDS   = [
    'o', 'a', 'os', 'as', 'de', 'do', 'da', 'dos', 'das', 
    'em', 'no', 'na', 'nos', 'nas', 'um', 'uma', 'uns', 'umas',
    'e', 'que', 'com', 'por', 'para', 'se', 'meu', 'seu', 'não'
]

def find_next_candidates(
        current_word: str, 
        conditional_probs: str,
        marginal_probs: Dict,
        words_frequencies: Dict,
        N: int,
        alpha: Optional[float]=1
    ) -> Dict:
    """
    Calcula a probabilidade de todas as palavras do vocabulário serem a próxima palavra.

    A função aplica uma variação do Teorema de Bayes para normalizar a probabilidade 
    de cada candidato, dado o contexto da 'current_word'. Utiliza a probabilidade 
    total para garantir que a soma das probabilidades dos candidatos seja 1.

    Args:
        current_word (str): A palavra atual na frase.
        conditional_probs (dict): Dicionário com P(Próxima|Atual) calculadas.
        marginal_probs (dict): Dicionário com a probabilidade a priori de cada palavra.
        words_frequencies (dict): Frequências acumuladas para o cálculo de suavização.
        N (int): Tamanho total do corpus/vocabulário.
        alpha (int, optional): Fator de suavização de Laplace. Padrão é 1.

    Returns:
        dict: Um dicionário onde as chaves são tuplas (current_word, candidato) 
              e os valores são as probabilidades normalizadas.
    """

    next_candidates_probs = {}
    next_candidates = marginal_probs.keys() 
    total_prob = total_probability(current_word, words_frequencies, conditional_probs, marginal_probs, N, alpha)

    for candidate in next_candidates:
        conditional_prob = conditional_probs[(current_word, candidate)] \
                            if   (current_word, candidate) in conditional_probs \
                            else (alpha / (words_frequencies.get(candidate, 0) + (alpha * N)))
        
        next_candidates_probs[(current_word, candidate)] = (conditional_prob * marginal_probs[candidate]) / total_prob
    
    return next_candidates_probs


def choose_next_word(next_candidates_probs: Dict) -> str:
    """
    Seleciona a próxima palavra utilizando amostragem ponderada sobre os melhores candidatos.

    A função extrai os 30 melhores candidatos. Com 50% de chance, ela filtra 
    'stop words' para gerar resultados mais semanticamente interessantes. 
    A escolha final é feita via random.choices, onde candidatos com maior 
    probabilidade têm proporcionalmente mais chance de serem escolhidos.

    Args:
        next_candidates_probs (dict): Dicionário gerado por find_next_candidates.

    Returns:
        str: A palavra selecionada para dar continuidade à frase.
    """
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