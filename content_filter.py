from enum import Enum
from PIL import Image, ImageFilter
import os

from shared import GUIFiltersLabels, IMG_DIR

BLURRED_IMG_CACHE_DIR = os.path.join(IMG_DIR, "blur_cache")
if not os.path.exists(BLURRED_IMG_CACHE_DIR):
    os.makedirs(BLURRED_IMG_CACHE_DIR)

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

    # For GUI gallery and question history 
    def is_rating_filtered_gui(self, gui_label: str, image_rating: str) -> bool:
        """Check if the image should be blurred based on GUI label and Image Rating."""
        content_filter_level = GUI_FILTER_LABELS_TO_IMAGE_RATINGS_MAPPING[gui_label]
        return RATING_LEVELS[image_rating] > RATING_LEVELS[content_filter_level]
    
    def get_blurred_image_from_path(self, image_path: str) -> Image.Image:
        """
        Fetches the blurred version of an image from the cache or generates it if not available.
        
        This function first checks a cache directory (`BLURRED_IMG_CACHE_DIR`) for a pre-computed
        blurred version of the image provided by `image_path`. If the blurred image is found,
        it's returned immediately, avoiding unnecessary computation.
        
        If the blurred version is not found in the cache, the function then blurs the original
        image, saves this blurred version to the cache directory, and returns the blurred image.
        
        Parameters:
        - image_path (str): The file path of the original image that needs to be blurred.

        Returns:
        - Image.Image: The blurred version of the original image.

        Example:
        >>> img = get_blurred_image_from_path("/path/to/original/image.png")
        """
        # Extract the filename from the original path
        filename = os.path.basename(image_path)
        blurred_image_path = os.path.join(BLURRED_IMG_CACHE_DIR, filename)
        
        # Check if the blurred version exists in the cache
        if os.path.exists(blurred_image_path):
            return Image.open(blurred_image_path)
        
        # If blurred version doesn't exist, create, save and return it
        image = Image.open(image_path)
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=50))
        
        blurred_image.save(blurred_image_path)
        
        return blurred_image
