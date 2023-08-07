import csv, os, json, cv2, numpy as np, io, random
from keras.models import load_model
from huggingface_hub import hf_hub_download
from PIL import Image, PngImagePlugin
from typing import Dict, Any, Tuple, List, Union

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
TAGS_RENAME_MAP_PATH = os.path.join(ROOT_DIR, 'wd14_tagging', 'tags_renaming.json')

MOCK_GEN_TAGS = [
    ('sky', 0.9), ('scenery', 0.88), ('star (sky)', 0.83), ('cloud', 0.81), ('starry sky',   
    0.76), ('whale', 0.75), ('bird', 0.67), ('night', 0.47), ('night sky', 0.41), ('sunset', 0.4), ('animal', 0.39), 
    ('shooting star', 0.37), ('cloudy sky', 0.35)
]
MOCK_GEN_FALSE_TAGS = [('cracked skin', 0.0), ('pink theme', 0.0), ('waving', 0.0), ('park bench', 0.0), ('afterimage', 0.0)]

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
                 remove_underscores_setting=True,
                 recursive=False,
                 debug=True,
                 force_download=False):
        
        self._repo_id = repo_id
        self._model_dir = model_dir
        self._batch_size = batch_size
        self._general_tag_threshold = general_tag_threshold
        self._character_tag_threshold = character_tag_threshold
        self._remove_underscores_setting = remove_underscores_setting
        self._recursive = recursive
        self._debug = debug
        self._force_download = force_download
        self._model = None
        self._tags_rating = None
        self._tags_general = None
        
        self.whitelisted_tags = None
        self.tag_rename_mappings = None
        self.event_handler = EventHandler()

        self._all_tags_weights = None
        self._all_general_tags_names_and_weights = None
        self._image_tags = None
        self._image_rating = None
        self._random_false_tags = None

    def _set_all_tags_weights_list(self, value):
        self._all_tags_weights = value

    @property
    def image_tags(self):
        return self._image_tags

    def _set_image_tags(self, value):
        self._image_tags = value

    @staticmethod
    def get_mock_tags():
        return MOCK_GEN_TAGS

    def _set_image(self, value):
        self._image = value

    @property
    def image_rating(self):
        return self._image_rating

    def _set_image_rating(self, value):
        self._image_rating = value

    @property
    def random_false_tags(self):
        return self._random_false_tags
    
    def _set_random_false_tags(self, value):
        self._random_false_tags = value

    @property
    def image_rating_and_tags(self):
        return {'rating': self.image_rating, 'tags': self.image_tags}

    def _load_whitelisted_tags(self):
        with open(TAGS_WHITELIST_PATH, 'r') as file:
            data = json.load(file)
        self.whitelisted_tags = data

    def _load_tags_rename_mappings(self):
        with open(TAGS_RENAME_MAP_PATH, 'r') as file:
            data = json.load(file)
        self.tag_rename_mappings = data

    def apply_tags_rename_mappings(self, tags_and_weights: List[Tuple[str, float]]):
        renamed_tags = []
        for tup in tags_and_weights:
            str_val, float_val = tup
            new_str = self.tag_rename_mappings.get(str_val, str_val)  # Get the new string value or keep the old one
            renamed_tags.append((new_str, float_val))
        return renamed_tags

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
        # self._tags_character = [row[1] for row in rows[1:] if row[2] == "4"]
    
    def _setup_tagger(self):
        self.event_handler.notify_observers(0.0, "Initializing WD14 Tagger")
        if not self.whitelisted_tags: self._load_whitelisted_tags()
        if not self.tag_rename_mappings: self._load_tags_rename_mappings()
        if not self._model: self._setup_model()
        if None in [self._tags_rating, self._tags_general]:
            self._setup_tags_list()
        self._clear_tags_and_rating()

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Convert the image to a NumPy array (float and range normalised to [0, 1])"""
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
        
    def _remove_underscores(self, input_data: Union[str, List[str], List[Tuple[str, float]]]) -> Union[str, List[str], List[Tuple[str, float]]]:
        """
        Removes underscores from input data. The function can handle string, list of strings and list of tuples.
        
        Parameters:
        input_data (Union[str, List[str], List[Tuple[str, float]]]): A string or a list of strings or list of tuples (str, float)
        
        Returns:
        Union[str, List[str], List[Tuple[str, float]]]: The input_data with underscores removed from strings
        """
        if not self._remove_underscores_setting: 
            return input_data
        
        def remove_from_string(tag_name: str) -> str:
            if len(tag_name) > 3:  # Remove undescores except for emojis e.g., ^_^, U_U
                return tag_name.replace("_", " ")
            return tag_name
        
        if isinstance(input_data, str):
            return remove_from_string(input_data)
        
        elif isinstance(input_data, list):
            if all(isinstance(i, str) for i in input_data):
                return [remove_from_string(tag_name) for tag_name in input_data]
            elif all(isinstance(i, tuple) and len(i) == 2 for i in input_data):
                return [(remove_from_string(tag_name), weight) for tag_name, weight in input_data]
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")

        return input_data

    def _process_tags(self, threshold) -> [str, float]:
        """Processes tags based on their weights, applying certain filters and thresholds."""
        accepted_tags = {}

        for tag_name, weight in self._all_general_tags_names_and_weights:
            if weight < threshold: 
                continue

            # Filter out any tags that are not in the whitelist
            if tag_name not in self.whitelisted_tags: 
                continue

            tag_name = self._remove_underscores(tag_name)

            accepted_tags[tag_name] = weight  # Create dict of tag names and weights

        return accepted_tags

    def _calculate_tags_weights(self, image_array: np.ndarray):
        self.event_handler.notify_observers(0.2, "Calculating tag weights")
        all_tag_weights_list = self._model(image_array, training=False)   # Use model to get tag weights from image numpy array
        all_tag_weights_list = all_tag_weights_list.numpy()   # Converts the tag weights tensor object into an numpy.ndarray object
        all_tag_weights_list = all_tag_weights_list[0]

        # Convert float32 to standard floats and round to two decimal places
        rounded_weights = [round(float(weight), 2) for weight in all_tag_weights_list]

        self._set_all_tags_weights_list(rounded_weights)

    def _zip_general_tags_names_and_weights(self):
        self._all_general_tags_names_and_weights = list(zip(self._tags_general, self._all_tags_weights[4:]))

    def _determine_image_rating(self) -> str:
        self.event_handler.notify_observers(0.3, "Determining image rating")
        img_rating_weights = zip(self._tags_rating, self._all_tags_weights[:4])    # Pair image rating tags with their corresponding weights
        image_rating = max(img_rating_weights, key=lambda x: x[1])[0]   # Set rating with highest weight as image rating

        self._image_rating = image_rating

    def _process_and_filter_image_tags(self):
        self.event_handler.notify_observers(0.5, "Processing and filtering tags")
        
        accepted_tags = self._process_tags(self._general_tag_threshold)

        # Sort the tags by weights and create a list of tuples
        sorted_tags_and_weights = sorted(accepted_tags.items(), key=lambda x: x[1], reverse=True)  

        self._set_image_tags(sorted_tags_and_weights)

        if self._debug: logger.info(f"WD14 Tagger: {self.image_rating_and_tags}")

    def add_wd14_metadata_to_image(self, image: Image.Image) -> Image.Image:
        img_tags_json = json.dumps(self.image_rating_and_tags)  # Convert the tags to a JSON-formatted string
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

    def _clear_tags_and_rating(self):
        self._all_tags_weights = None
        self._image_tags = None
        self._image_rating = None
        self._random_false_tags = None

    def generate_random_false_tags(self, count: int) -> List[Tuple[str, float]]:
        """
        Generate a list of unique tuples with renamed tags and their associated weights.
        
        This function works in the following steps:
        1. Pre-filters the _all_tags_weights list to include only those tags that are in whitelisted_tags.
        2. Randomly selects tags from the whitelisted_tags list, ensuring previously sampled tags are not re-selected.
        3. Renames the tags according to the rename_mapping.
        4. Adds the renamed tag and its weight to the result, ensuring that:
            - The renamed tag doesn't already exist in _image_tags.
            - The resulting tuple is unique.
        5. Repeats the process until the desired number of tuples (specified by 'count') is obtained or all tags are processed.

        Parameters:
        - count (int): The number of unique tuples to generate.

        Returns:
        - list of tuples: A list containing unique tuples of the form (renamed_tag, weight).

        Note:
        - If the desired count can't be achieved (for instance, if there aren't enough tags in whitelisted_tags or too many tags 
            are already in _image_tags), the function might return fewer tuples than requested.
        """
        result_tuples = set()
        already_sampled_tags = set()
        
        # Pre-filtered list to improve efficiency
        filtered_tags_weights = [t for t in self._all_general_tags_names_and_weights if t[0] in self.whitelisted_tags]
        
        # As long as we don't have enough result tuples
        while len(result_tuples) < count:
            remaining_tags = [tag for tag in self.whitelisted_tags if tag not in already_sampled_tags]
            
            # If there are no more remaining tags, you might need to reconsider the approach or inputs
            if not remaining_tags:
                break
            
            # Shuffle the remaining tags and chunk them
            random.shuffle(remaining_tags)
            chunk = remaining_tags[:count - len(result_tuples)]
            
            # Add to already sampled set
            already_sampled_tags.update(chunk)
            
            temp_tuples = [t for t in filtered_tags_weights if t[0] in chunk]

            for t in temp_tuples:
                renamed_tag = self.apply_tags_rename_mappings([t])
                renamed_tag = renamed_tag[0]

                if renamed_tag not in self._image_tags and (renamed_tag, t[1]) not in result_tuples:
                    result_tuples.add(renamed_tag)

        result_tuples = list(result_tuples) # set to list
        result_tuples = self._remove_underscores(result_tuples) # update with new names

        self._set_random_false_tags(result_tuples)

        if self._debug: logger.info(f"False tags: {self.random_false_tags}")

    @log_function_call
    def generate_tags(self, image: Image.Image):
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
        image_array = self._preprocess_image(image) 
        self._calculate_tags_weights(image_array)
        self._zip_general_tags_names_and_weights()
        self._determine_image_rating()
        self._process_and_filter_image_tags()
        self._set_image_tags(self.apply_tags_rename_mappings(self.image_tags))

    def mock_generate_tags(self):
        """ For debugging. """
        self._image_rating = 'general'
        self._set_image_tags(MOCK_GEN_TAGS)

    def mock_gen_false_tags(self):
        self._set_random_false_tags(MOCK_GEN_FALSE_TAGS)
