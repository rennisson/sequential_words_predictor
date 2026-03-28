from collections import Counter
from .time_measurement import time_measurement
from typing import Dict, List, Tuple

@time_measurement
def words_frequencies_map(text: List) -> Dict:
    """
    Mapeia a frequência de ocorrência de pares de palavras (bigramas).

    Percorre a lista de palavras e conta quantas vezes cada combinação 
    consecutiva de (palavra_atual, proxima_palavra) aparece no texto.

    Args:
        text (list): Lista de strings representando o corpo do texto processado.

    Returns:
        dict: Dicionário onde as chaves são tuplas (atual, proxima) e os 
              valores são as contagens inteiras.
    """

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
def marginal_probabilities(text: List) -> Dict[str, float]:
    """
    Calcula a probabilidade individual (a priori) de cada palavra no texto.

    Baseia-se na frequência absoluta de cada palavra dividida pelo número 
    total de palavras no corpus (N).

    Args:
        text (list): Lista de strings contendo todas as palavras do texto.

    Returns:
        dict: Dicionário com cada palavra como chave e sua probabilidade 
              marginal (float entre 0 e 1) como valor.
    """

    print("Calculating marginal probabilities...")

    N = len(text)
    print(f"{N=}")
    words_counter = Counter(text)
    marginal_probs = {key: freq / N for key, freq in words_counter.items()}

    return marginal_probs

@time_measurement
def conditional_probabilities(words_map: Dict, N: int, alpha: float=1.0) -> Tuple[Dict, Dict]:
    """
    Calcula as probabilidades condicionais P(próxima|atual) com suavização.

    Utiliza o método de Laplace Smoothing (alpha) para garantir que 
    combinações não vistas não resultem em probabilidade zero.

    Args:
        words_map (dict): Dicionário de frequências de bigramas.
        N (int): Tamanho total do vocabulário ou do corpus.
        alpha (int, optional): Fator de suavização. Padrão é 1.

    Returns:
        tuple: (conditional_probs, words_frequencies)
            - conditional_probs (dict): P(B|A) para cada par de palavras.
            - words_frequencies (dict): Soma das frequências de cada palavra 
              na posição de 'próxima'.
    """

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


def total_probability(current_word, words_frequencies, conditional_probs, marginal_probs, N, alpha=1) -> float:
    """
    Calcula a probabilidade total de uma palavra no contexto atual do modelo.

    Aplica o Teorema da Probabilidade Total somando o produto da 
    probabilidade condicional pela probabilidade marginal de cada palavra.

    Args:
        current_word (str): A palavra base para a predição.
        words_frequencies (dict): Frequências acumuladas de palavras sucessoras.
        conditional_probs (dict): Mapa de probabilidades P(B|A).
        marginal_probs (dict): Mapa de probabilidades P(A).
        N (int): Total de palavras do corpus.
        alpha (int, optional): Fator de suavização. Padrão é 1.

    Returns:
        float: O valor da probabilidade total calculada.
    """

    total_probability = 0
    for word in marginal_probs.keys():
        conditional_prob = conditional_probs[(current_word, word)] \
                            if   (current_word, word) in conditional_probs \
                            else (alpha / (words_frequencies.get(word, 0) + (alpha * N)))
        
        priori = marginal_probs[word]
        total_probability += conditional_prob * priori

    return total_probability