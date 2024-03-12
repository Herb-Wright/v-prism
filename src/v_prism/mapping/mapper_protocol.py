from typing import Protocol

from torch import Tensor


class MapperProtocol(Protocol):
    def update(self, X: Tensor, y: Tensor) -> None:
        """updates map with new data
        
        Args:
            X: (P, 3) float array of x values for new points
            y: array of y values for new points
        """
        raise NotImplementedError()
    
    def predict(self, X: Tensor) -> Tensor:
        """predicts y values for given X array
        
        
        """
        raise NotImplementedError()





