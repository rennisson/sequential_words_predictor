from pathlib import Path
from typing import Any, Callable, Dict, Union
import pickle


def get_file(path: Union[str, Path]) -> Any:
    """
    Carrega um objeto Python a partir de um arquivo binário (Pickle).

    Lê o arquivo no caminho especificado e reconstrói o objeto original 
    (dicionário, lista, etc.) mantendo sua estrutura de dados.

    Args:
        path (str|Path): O caminho completo para o arquivo .pkl.

    Returns:
        object: O objeto Python recuperado do arquivo.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_file(path: Union[str, Path], object: Callable):
    """
    Serializa e salva um objeto Python em um arquivo binário.

    Utiliza o módulo pickle para converter o objeto em um fluxo de bytes 
    e gravá-lo no disco para persistência.

    Args:
        path (str|Path): O caminho de destino para o arquivo .pkl.
        object (Callable): O objeto Python (dicionário, int, lista) a ser salvo.
    """
    with open(path, "wb") as f:
        pickle.dump(object, f)


def save_model(
        marginal_probs: Dict,
        N: int,
        conditional_probs: Dict,
        words_frequencies: Dict):
    """
    Exporta os quatro componentes essenciais do modelo para a pasta 'pkl_files'.

    Esta função organiza a persistência do modelo treinado, salvando as 
    probabilidades condicionais, marginais, as frequências acumuladas e o 
    tamanho do vocabulário em arquivos separados. Os arquivos são armazenados 
    em um diretório pai relativo à localização do script.

    Args:
        marginal_probs (dict): Dicionário de probabilidades a priori.
        N (int): Tamanho total do vocabulário.
        conditional_probs (dict): Mapa de probabilidades P(B|A).
        words_frequencies (dict): Contagem de frequências de sucessores.
    """
    print(f"Saving model...")

    folder_path = Path(__file__).parent.parent / "pkl_files"

    file_path = folder_path / "cond_probs.pkl"
    save_file(file_path, conditional_probs)

    file_path = folder_path / "words_frequencies.pkl"
    save_file(file_path, words_frequencies)

    file_path = folder_path / "marginal_probs.pkl"
    save_file(file_path, marginal_probs)

    file_path = folder_path / "len_vocabulary.pkl"
    save_file(file_path, N)
