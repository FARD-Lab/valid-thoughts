from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator

class BaseLoader(ABC):

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_text_field(self, example: Dict[str, Any]) -> str:
        pass

    @abstractmethod
    def reject_example(self, example: Dict[str, Any]):
        pass

    @property
    @abstractmethod
    def instruction(self) -> str:
        pass
