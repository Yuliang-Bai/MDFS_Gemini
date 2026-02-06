from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np

@dataclass
class FitResult:
    """统一的模型输出结果"""
    selected_features: Dict[str, List[int]]
    model_state: Optional[Dict[str, Any]] = None

class BaseMethod(ABC):
    """所有方法的抽象基类"""
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.params = kwargs

    @abstractmethod
    def fit(self, X: Dict[str, np.ndarray], y: Optional[np.ndarray] = None) -> FitResult:
        pass
