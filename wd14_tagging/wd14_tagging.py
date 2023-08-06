import csv, os, json, cv2, numpy as np, io
from keras.models import load_model
from huggingface_hub import hf_hub_download
from PIL import Image, PngImagePlugin
from typing import Dict, Any

from custom_logging import logger, log_function_call
from shared import ROOT_DIR, EventHandler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow debug & info (build for CPU optimization etc.)

# from wd14 tagger
IMAGE_SIZE = 448

# wd-v1-4-swinv2-tagger-v2 / wd-v1-4-vit-tagger / wd-v1-4-vit-tagger-v2/ wd-v1-4-convnext-tagger / wd-v1-4-convnext-tagger-v2
DEFAULT_WD14_TAGGER_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
DEFAULT_WD14_PATH = os.path.join(ROOT_DIR, 'library', 'wd-v1-4-convnext-tagger-v2')
FILES = ["keras_metadata.pb", "saved_model.pb", "selected_tags.csv"]
SUB_DIR = "variables"
SUB_DIR_FILES = ["variables.data-00000-of-00001", "variables.index"]
CSV_FILE = FILES[-1]
TAGS_WHITELIST_PATH = os.path.join(ROOT_DIR, 'wd14_tagging', 'tags_whitelist.json')

class WD14Tagger():
    """
    Parameters:
    - repo_id (str, default=DEFAULT_WD14_TAGGER_REPO): Identifier for the WD14 tagger repository on Hugging Face.
    - model_dir (str, default="wd14_tagger_model"): Directory to store the WD14 tagger model.
    - batch_size (int, default=8): Number of samples per gradient update.
    - general_tag_threshold (float, default=0.35): Threshold of confidence to add a tag for the general category.
    - character_tag_threshold (float, default=0.35): Threshold of confidence to add a tag for the character category.
    - remove_underscores (bool, default=True): If True, replaces underscores with spaces in the output tags.
    - recursive (bool, default=False): If True, searches for images in subfolders recursively.
    - debug (bool, default=True): If True, enables debug logging.
    - force_download (bool, default=False): If True, forces downloading of the WD14 tagger models.
    """
    def __init__(self,
                 repo_id=DEFAULT_WD14_TAGGER_REPO,
                 model_dir=DEFAULT_WD14_PATH,
                 batch_size=8,
                 general_tag_threshold=0.35,
                 character_tag_threshold=0.35,
                 remove_underscores=True,
                 recursive=False,
                 debug=True,
                 force_download=False):
        
        self._repo_id = repo_id
        self._model_dir = model_dir
        self._batch_size = batch_size
        self._general_tag_threshold = general_tag_threshold
        self._character_tag_threshold = character_tag_threshold
        self._remove_underscores = remove_underscores
        self._recursive = recursive
        self._debug = debug
        self._force_download = force_download
        self._model = None
        self._tags_rating = None
        self._tags_general = None
        self._tags_character = None
        
        self.whitelisted_tags = None
        self.tag_rename_mappings = None
        self.event_handler = EventHandler()

    def _load_whitelisted_tags(self):
        with open(TAGS_WHITELIST_PATH, 'r') as file:
            data = json.load(file)
        self.whitelisted_tags = data

    def _load_tags_new_names(self):
        with open(TAGS_WHITELIST_PATH, 'r') as file:
            data = json.load(file)
        self.tag_rename_mappings = data

    def _setup_model(self):
        """
        Sets up and initializes the WD14 tagger model.

        If the model directory doesn't exist or if force_download is set to True, this method
        downloads the necessary files for the WD14 tagger model from the specified Hugging Face
        repository. The model is then loaded from the local directory.

        Behavior:
        - If the model directory does not exist or if self.force_download is True, the method
        downloads the WD14 tagger model from the Hugging Face repository identified by self.repo_id.
        - Files are downloaded into the directory specified by self.model_dir.
        - If the model directory already exists and self.force_download is False, the method simply
        loads the model from the local directory.

        Usage:
        img_tagger = ImageTagger()
        img_tagger.setup_model()
        """
        if self._model: return
        if not os.path.exists(self._model_dir) or self._force_download:
            logger.info(f"Downloading WD14 tagger model from hf_hub, Repo ID: {self._repo_id}")
            for file in FILES:
                hf_hub_download(self._repo_id, file, cache_dir=self._model_dir, force_download=True, force_filename=file)
            for file in SUB_DIR_FILES:
                hf_hub_download(
                    self._repo_id,
                    file,
                    subfolder=SUB_DIR,
                    cache_dir=os.path.join(self._model_dir, SUB_DIR),
                    force_download=True,
                    force_filename=file,
                )

        self._model = load_model(self._model_dir, compile=False)
        
    def _setup_tags_list(self):
        """
        Sets up the lists of rating, general, and character tags by reading them from a CSV file
        located in the model directory.

        This method reads the CSV file specified by CSV_FILE from the model directory, and parses
        the tags into three categories: rating, general, and character. The tags are then stored
        as class attributes.

        The CSV file must have the following header: ['tag_id', 'name', 'category', 'count']
        and must contain rows representing the tags. The method filters the tags into categories
        based on the value in the 'category' column.

        Attributes Created:
        - self.tags_rating: List of rating tags.
        - self.tags_general: List of general tags.
        - self.tags_character: List of character tags.

        Raises:
        - AssertionError: If the CSV file's header does not match the expected format.

        Usage:
        img_tagger = ImageTagger()
        img_tagger.setup_tags_list()
        """
        # label_names = pd.read_csv(os.path.join(model_dir, CSV_FILE))
        with open(os.path.join(self._model_dir, CSV_FILE), "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            l = [row for row in reader]
            header = l[0]  # tag_id, name, category, count
            rows = l[1:]
        assert header[0] == "tag_id" and header[1] == "name" and header[2] == "category", f"unexpected csv format: {header}"


        self._tags_rating = [row[1] for row in rows[0:] if row[2] == "9"]
        self._tags_general = [row[1] for row in rows[1:] if row[2] == "0"]
        self._tags_character = [row[1] for row in rows[1:] if row[2] == "4"]
    
    def _setup_tagger(self):
        self.event_handler.notify_observers(0.0, "Initializing WD14 Tagger")
        if not self.whitelisted_tags: self._load_whitelisted_tags()
        if not self._model: self._setup_model()
        if None in [self._tags_rating, self._tags_general, self._tags_character]:
            self._setup_tags_list()

    @staticmethod
    def _collate_fn_remove_corrupted(batch):
        """Collate function that allows to remove corrupted examples in the
        dataloader. It expects that the dataloader returns 'None' when that occurs.
        The 'None's in the batch are removed.
        """
        # Filter out all the Nones (corrupted examples)
        batch = list(filter(lambda x: x is not None, batch))
        return batch

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        self.event_handler.notify_observers(0.1, "Pre-processing generated image")
        image = image.convert("RGB")    # Converting to "RGB" mode ensures that image has exactly 3 channels
        image = np.array(image)
        image = image[:, :, ::-1]  # RGB->BGR

        # pad to square
        size = max(image.shape[0:2])
        pad_x = size - image.shape[1]
        pad_y = size - image.shape[0]
        pad_l = pad_x // 2
        pad_t = pad_y // 2
        image = np.pad(image, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode="constant", constant_values=255)

        interp = cv2.INTER_AREA if size > IMAGE_SIZE else cv2.INTER_LANCZOS4
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interp)

        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)   # Expand dimensions to create a batch with a single image

        return image

    def _process_tags(self, tag_weights, start_index, tag_names, threshold) -> Dict[str, float]:
        """
        Processes a set of tags based on their weights, applying certain filters and thresholds.
        This function is responsible for filtering out undesired tags, replacing underscores in tags,
        rounding the weights to two decimal places, and constructing a dictionary of accepted tag names
        and weights.

        Parameters:
        - tag_weights (array-like): The weights associated with each tag, likely obtained from a trained model.
        - start_index (int): The starting index in tag_weights from which to process the tags.
        - tag_names (list): A list of tag names corresponding to the weights in tag_weights.
        - threshold (float): The minimum acceptable weight for a tag to be included in the result.

        Returns:
        - Dict[str, float]: A dictionary containing the accepted tags and their corresponding weights,
                            filtered, formatted, and sorted based on the given criteria.

        Behavior:
        - Ignores tags with weights below the given threshold.
        - Replaces underscores in tag names with spaces, except for certain patterns like emojis (e.g., "^_^").
        - Rounds the weights to two decimal places.

        Example usage:
        accepted_general_tags = obj.process_tags(tag_weights, 0, general_tag_names, 0.5)
        """
        accepted_tags = {}

        for i, weight in enumerate(tag_weights[start_index:]):
            if i >= len(tag_names): break
            if weight < threshold: continue

            tag_name = tag_names[i]

            if tag_name not in self.whitelisted_tags: 
                continue

            if self._remove_underscores and len(tag_name) > 3:   # Remove undescores except for emojis e.g., ^_^, U_U
                tag_name = tag_name.replace("_", " ")

            weight = round(float(weight), 2)    # Convert float32 to standard floats and round to two decimal places
            accepted_tags[tag_name] = weight    # Create dict of tag names and rounded weights

        return accepted_tags

    def _calculate_image_tags_weights(self, image_array: np.ndarray):
        self.event_handler.notify_observers(0.2, "Calculating tag weights")
        img_tag_weights = self._model(image_array, training=False)   # Use model to get tag weights from image numpy array
        img_tag_weights = img_tag_weights.numpy()   # Converts the tag weights tensor object into an numpy.ndarray object
        img_tag_weights = img_tag_weights[0]
        return img_tag_weights
    
    def _get_image_rating(self, img_tags_weights: np.ndarray) -> str:
        self.event_handler.notify_observers(0.3, "Determining image rating")
        img_rating_weights = zip(self._tags_rating, img_tags_weights[:4])    # Pair image rating tags with their corresponding weights
        image_rating = max(img_rating_weights, key=lambda x: x[1])[0]   # Set rating with highest weight as image rating
        return image_rating

    def _process_and_filter_image_tags(self, img_tags_weights: np.ndarray, img_rating: str) -> Dict[str, Any]:
        self.event_handler.notify_observers(0.5, "Processing and filtering tags")
        
        accepted_tags = self._process_tags(
            img_tags_weights, len(self._tags_rating), self._tags_general, self._general_tag_threshold,
        )

        # Sort the tags by weights and create a list of tuples
        sorted_tags_and_values = sorted(accepted_tags.items(), key=lambda x: x[1], reverse=True)  

        tags_with_img_rating = {'rating': img_rating, 'tags': sorted_tags_and_values}

        if self._debug: logger.info(f"WD14 Tagger: {tags_with_img_rating}")

        return tags_with_img_rating

    def _add_tags_to_image_metadata(self, image: Image.Image, img_tags):
        img_tags_json = json.dumps(img_tags)  # Convert the tags to a JSON-formatted string
        self.event_handler.notify_observers(0.8, "Adding tags to image metadata")
        pnginfo = PngImagePlugin.PngInfo()  # Create a PngInfo object

        for key, value in image.info.items():   # Get the existing metadata (if any) and add to PngInfo
            if key != 'dpi':
                pnginfo.add_text(key, value)

        pnginfo.add_text("wd14_tagger", img_tags_json) # Add the tags JSON string as a "wd14_tagger" parameter

        output = io.BytesIO()   # Create a bytes buffer to hold the new image data
        image.save(output, format="PNG", pnginfo=pnginfo)
        image = Image.open(output)  # Convert bytes back to a PIL Image object

        return image

    @log_function_call
    def tag_image(self, image: Image.Image):
        """
        Parameters:
        - image (Image.Image): The input image that needs to be tagged.

        Returns:
        - image (Image.Image): The same input image with the tags added to its metadata.
        - img_tags (str): The JSON-formatted string representing the extracted tags, including
        rating, general, and character tags.

        Usage:
        img_tagger = ImageTagger()
        tagged_image, tags = img_tagger.tag_image(image)
        """
        self._setup_tagger()
        image_array = self._preprocess_image(image) # Convert the image to a NumPy array (float and range normalised to [0, 1])
        img_tag_weights = self._calculate_image_tags_weights(image_array)
        img_rating = self._get_image_rating(img_tag_weights)
        img_tags = self._process_and_filter_image_tags(img_tag_weights, img_rating)
        image = self._add_tags_to_image_metadata(image=image, img_tags=img_tags)

        return image, img_tags
