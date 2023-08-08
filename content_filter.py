from enum import Enum
from PIL import Image, ImageFilter

from shared import GUIFiltersLabels

class ImageRatings(Enum):
    GENERAL = 'general'
    SENSITIVE = 'sensitive'
    QUESTIONABLE = 'questionable'
    EXPLICIT = 'explicit'

RATING_LEVELS = {
    ImageRatings.GENERAL.value: 1, 
    ImageRatings.SENSITIVE.value: 2, 
    ImageRatings.QUESTIONABLE.value: 3, 
    ImageRatings.EXPLICIT.value: 4
}

GUI_FILTER_LABELS_TO_IMAGE_RATINGS_MAPPING = {
    GUIFiltersLabels.GENERAL.value: ImageRatings.GENERAL.value,
    GUIFiltersLabels.SENSITIVE.value: ImageRatings.SENSITIVE.value,
    GUIFiltersLabels.QUESTIONABLE.value: ImageRatings.QUESTIONABLE.value,
    GUIFiltersLabels.EXPLICIT.value: ImageRatings.EXPLICIT.value
}

IMAGE_RATING_TO_GUI_FILTER_MAPPING = {v: k for k, v in GUI_FILTER_LABELS_TO_IMAGE_RATINGS_MAPPING.items()}

class ContentFilter():
    def __init__(self):
        self._content_filter_level=None
        self._original_image=None

    @property
    def original_image(self):
        return self._original_image

    def _set_original_image(self, img: Image.Image):
        self._original_image = img

    def set_content_filter_level(self, content_filter_level: str):
        img_rating = GUI_FILTER_LABELS_TO_IMAGE_RATINGS_MAPPING[content_filter_level]
        self._content_filter_level = RATING_LEVELS[img_rating]

    def get_content_filter_level(self) -> str:
        gui_label = IMAGE_RATING_TO_GUI_FILTER_MAPPING[self._content_filter_level]
        return gui_label

    def is_rating_filtered(self, image_rating: str) -> bool:
        return RATING_LEVELS[image_rating] > self._content_filter_level

    def blur_image(self, image: Image.Image):
        self._set_original_image(image) # Save unblurred image
        return image.filter(ImageFilter.GaussianBlur(radius=50))
