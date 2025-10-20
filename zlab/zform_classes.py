from dataclasses import dataclass
from typing import Callable, Sequence

@dataclass
class ZformModel:
    name: str
    func: Callable
    init_guess: Callable[[np.ndarray, np.ndarray], Sequence[float]]
    bounds: Callable[[np.ndarray, np.ndarray], tuple[Sequence[float], Sequence[float]]]
    higher_is_better: bool = True  # optional per metrica

