from PIL import Image, ImageFilter

from shared import ImageRatings

DEFAULT_CONTENT_FILTER_LEVEL = 1

class ContentFilter():
    def __init__(self):
            self._content_filter_level=DEFAULT_CONTENT_FILTER_LEVEL

    def set_content_filter_level(self, level: int):
        self._content_filter_level = level

    def get_content_filter_level(self) -> int:
        return self._content_filter_level

    def is_rating_filtered(self, image_rating: str) -> bool:
        rating_levels = {
            ImageRatings.GENERAL: 1, 
            ImageRatings.SENSITIVE: 2, 
            ImageRatings.QUESTIONABLE: 3, 
            ImageRatings.EXPLICIT: 4
        }
        return rating_levels[image_rating] > self._content_filter_level

    def get_blurred_image(self, image: Image.Image):
        return image.filter(ImageFilter.GaussianBlur(radius=50))
