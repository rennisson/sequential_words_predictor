from functools import wraps
from typing import Callable
import time

def time_measurement(func: Callable) -> Callable:
    """
    Decorador para medir e exibir o tempo de execução de uma função.

    Captura o tempo de início e fim da execução da função decorada utilizando 
    time.perf_counter() para garantir alta precisão. Ao final, imprime no 
    console o nome da função e o tempo total decorrido em segundos.

    Args:
        func (callable): A função que será envolvida (wrapped) pelo decorador.

    Returns:
        callable: Uma versão modificada da função original que inclui a 
                  medição de tempo, mantendo os metadados originais (via @wraps).
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[{func.__name__}] Execution time: {end - start:.4f}s")
        return result
    return wrapper 