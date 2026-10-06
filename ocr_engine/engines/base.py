from abc import ABC, abstractmethod

from PIL import Image


class OCREngine(ABC):
    """Common interface: every engine takes a PIL image and returns plain text."""

    name: str = "base"

    @abstractmethod
    def recognize(self, image: Image.Image) -> str:
        ...
