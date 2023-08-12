import asyncio, aiohttp, random, re, os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from math import floor
from typing import List, Tuple

from custom_logging import logger
from sd_api_wrapper import generate_image as sd_generate_image, get_progress, mock_generate_image
from shared import (SDModels, EventHandler, RANDOM_MODEL_OPT_STRING, ObserverContext, DifficultyLevels, 
                    DIFFICULTY_LEVEL_TAG_RATIO, DIFFICULTY_LEVEL_EXP_GAIN, IMG_DIR)
from wd14_tagging.wd14_tagging import WD14Tagger
from content_filter import ContentFilter
from user_feedback import UserFeedback
from danbooru_api_wrapper import DanbooruApi
from db_manager import DatabaseManager, tags_to_json, json_to_tags

DEBUG_MOCK_GEN_INFO = [
    'Steps: 30',
    'Sampler: DPM++ 2M Karras',
    'CFG scale: 7.0',
    'Seed: 2929592300',
    'Size: 512x512',
    'Model hash: 40d4f9d626',
    'Model: sardonyxREDUX_v20',
    'Seed resize from: -1x-1',
    'Denoising strength: 0.45',
    'Clip skip: 2',
    'Version: v1.2.0'
]

class App:
    def __init__(self, debug_mock_image=False, debug_mock_tags=False):
        self._image = None
        self._image_gen_info = None
        self._image_gen_time = None
        self._image_is_filtered = False
        self._difficulty_level = None
        self._submitted_tags = None

        self._correct_answers_cache = None
        self._incorrect_answers_cache = None
        self._missed_answers_cache = None
        self._net_points_cache = None

        self.event_handler = EventHandler()
        self.wd14_tagger = WD14Tagger()
        self.content_filter = ContentFilter()
        self.danbooru_api = DanbooruApi()
        self.db_manager = DatabaseManager()
        self.user_feedback = UserFeedback()

        # Set function pointers based on the debug flags
        self._generate_image_func = self._generate_image if not debug_mock_image else self._mock_gen_image
        self._generate_tags_func = self._generate_tags if not debug_mock_tags else self._mock_gen_tags

    def _clear_round_data(self):
        self._image = None
        self._image_gen_info = None
        self._image_gen_time = None
        self._image_is_filtered = None
        self._difficulty_level = None
        self._submitted_tags = None

        self._correct_answers_cache = None
        self._incorrect_answers_cache = None
        self._missed_answers_cache = None
        self._net_points_cache = None

    def _run_sd_generate(self, prompt: str, gui_model_name: str):
        """
        Generates an image based on a specified prompt using a given model checkpoint,
        and periodically reports the progress of the generation.

        :param prompt: The prompt to use for image generation.
        :param checkpoint: The model checkpoint to use for image generation.
        :param gr_progress: A callable object that takes two parameters, the current progress
                            as a float, and a description string, which is used to report
                            progress back to the caller.
        :return: The generated image with metadata.
        """
        async def inner():
            sd_model = (
                SDModels.get_sd_model(gui_model_name) 
                if gui_model_name != RANDOM_MODEL_OPT_STRING 
                else SDModels.get_sd_model(
                    random.choice(SDModels.get_gui_model_names_list())
                )
            )
            self.event_handler.notify_observers(0.0, f'Loading model {SDModels.get_sd_model_name(sd_model)}  ...')
            
            async with aiohttp.ClientSession() as session:  # Run the async function and wait for it to complete
                post_task = asyncio.create_task(sd_generate_image(sd_model=sd_model, prompt=prompt))    # Create the POST task for the image request
                while not post_task.done():
                    progress, eta, current_step, total_steps, current_image = await get_progress(session)
                    self.event_handler.notify_observers(progress, f'Generating image - ETA: {eta:.2f}s - Step: {current_step}/{total_steps}')
                    await asyncio.sleep(1)
                self.event_handler.notify_observers(1.0, 'Finished generating image')
                image = await post_task   # Get the result of the POST task
                return image

        def run_async_code():
            loop = asyncio.new_event_loop() # Create a new event loop
            asyncio.set_event_loop(loop)    # Set the new event loop as the default loop for the new thread
            try:
                return loop.run_until_complete(inner()) # Run the inner coroutine inside the newly created event loop
            finally:
                loop.close()    # Close the event loop when finished

        with ThreadPoolExecutor() as executor:  # Run the asynchronous code inside a thread
            future = executor.submit(run_async_code)
            return future.result()
    
    def _run_wd14_tagger(self):
        with ObserverContext(self.wd14_tagger.event_handler, self.event_handler.get_latest_observer()):
            self.wd14_tagger.generate_tags(self.image)

    def _set_image(self, img: Image.Image):
        self._image = img

    def _set_image_generation_info(self, sd_gen_info):
        self._image_gen_info = sd_gen_info

    def _set_image_generation_time(self):
        self._image_gen_time = datetime.now()

    def _extract_image_generation_info(self):
        data = self.image.info.items()
        data_string = ' '.join(map(str, data)).strip("')")  # Convert tuple into string

        parameters = [
            'Steps', 'Sampler', 'CFG scale', 'Seed', 'Size', 
            'Model hash', 'Model', 'Seed resize from', 
            'Denoising strength', 'Clip skip', 'Version'
        ]

        extracted_values = []

        for param in parameters:
            match = re.search(f'{param}: ([^,]*)', data_string)
            if match:
                extracted_values.append(f'{param}: {match.group(1)}')

        self._set_image_generation_info(extracted_values)

    def _set_difficulty_level(self, d_level: str):
        self._difficulty_level = d_level

    @property
    def difficulty_level(self):
        return self._difficulty_level

    @property
    def image_gen_info(self):
        return ', '.join(self._image_gen_info)

    @property
    def image_gen_time(self):
        return self._image_gen_time

    @property
    def image_is_filtered(self):
        return self._image_is_filtered

    @property
    def image_tags(self):
        return self.wd14_tagger.image_tags
    
    @property
    def image_rating(self):
        return self.wd14_tagger.image_rating
    
    @property
    def false_tags(self):
        return self.wd14_tagger.random_false_tags

    @property
    def image(self):
        return self._image

    @property
    def original_image(self):
        return self.content_filter.original_image

    @property
    def tags_to_display(self):
        combined_tags = self.image_tags + self.false_tags
        random.shuffle(combined_tags)
        return combined_tags

    def _set_submitted_tags(self, tag_names_list: list):
        self._submitted_tags = tag_names_list

    @property
    def gained_points(self):
        if self._net_points_cache is None:
            # Define the functions using tag weight threshold from wd14 tagger
            gained_points, lost_points = self.create_points_calculators()
            # Calculate total points for correct answers
            total_gained = sum(gained_points(tag[1]) for tag in self.correct_answers)
            
            # Calculate total points lost for incorrect answers
            total_lost = sum(lost_points(tag[1]) for tag in self.incorrect_answers)
            
            # Calculate the net points, round down, min is 0
            net_points = total_gained + total_lost
            net_points = round(float(net_points), 1)
            net_points = max(net_points, 0)
            self._net_points_cache = net_points
        return self._net_points_cache

    @property
    def gained_exp(self):
        if self.gained_points <= 0:
            return 0

        difficulty_enum = DifficultyLevels[self.difficulty_level.upper()]
        exp_gain_multiplier = DIFFICULTY_LEVEL_EXP_GAIN[difficulty_enum]
        exp_gained = (self.gained_points * exp_gain_multiplier) * 10
        return int(exp_gained)
    
    @property
    def correct_answers(self):
        if self._correct_answers_cache is None:
            self._correct_answers_cache = [tag for tag in self.image_tags if tag[0] in self._submitted_tags]
        return self._correct_answers_cache
        
    @property
    def incorrect_answers(self):
        if self._incorrect_answers_cache is None:
            self._incorrect_answers_cache = [tag for tag in self.false_tags if tag[0] in self._submitted_tags]
        return self._incorrect_answers_cache

    @property
    def missed_answers(self):
        if self._missed_answers_cache is None:
            self._missed_answers_cache = [tag for tag in self.image_tags if tag[0] not in self._submitted_tags]
        return self._missed_answers_cache

    @staticmethod
    def tag_names_only(tags_list: List[Tuple[str, int]]) -> List[str]:
        return [t[0] for t in tags_list]

    def _generate_image(self, prompt: str, checkpoint: str):
        logger.info(f'Generating image with prompt: {prompt}')
        generated_image = self._run_sd_generate(prompt, checkpoint)
        self._set_image_generation_time()
        self._set_image(generated_image)
        self._extract_image_generation_info()

    def _generate_tags(self):
        self._run_wd14_tagger()
        self._set_image(self.wd14_tagger.add_wd14_metadata_to_image(self.image))
        self._generate_false_tags()
        
    def _generate_false_tags(self):
        f_tags_count = self._calculate_false_tags_count_based_on_difficulty_level_()
        self.wd14_tagger.generate_random_false_tags(f_tags_count)
        logger.info(f'Image has {len(self.image_tags)} tags. Generated {f_tags_count} false tags. Difficulty: {self.difficulty_level}')

    def _calculate_false_tags_count_based_on_difficulty_level_(self):
        difficulty_enum = DifficultyLevels[self.difficulty_level.upper()]
        f_tag_count = len(self.image_tags) * DIFFICULTY_LEVEL_TAG_RATIO[difficulty_enum]
        f_tag_count_rounded_down = floor(f_tag_count)
        return max(f_tag_count_rounded_down, 1) # max for at least 1 false tag (if img has 1 true tag)

    def _apply_image_rating_filter(self, content_filter_level: str):
        self.event_handler.notify_observers(0.9, 'Checking image rating against filters ...')
        self.content_filter.set_content_filter_level(content_filter_level)
        if self.content_filter.is_rating_filtered(self.image_rating):
            self._set_image(self.content_filter.blur_image(self.image))
            self._image_is_filtered = True  # flag current image as rating filtered
        
    def _mock_gen_image(self, *args):
        self._set_image(mock_generate_image())    # get mock image for debugging
        self._set_image_generation_info(DEBUG_MOCK_GEN_INFO)
        self._set_image_generation_time()

    def _mock_gen_tags(self, *args):
        self.wd14_tagger.mock_generate_tags()
        self.wd14_tagger.mock_gen_false_tags()

    def generate_round(self, prompt: str, checkpoint: str, content_filter_level: str, difficulty: str):
        self._clear_round_data()
        self._set_difficulty_level(difficulty)
        self._generate_image_func(prompt, checkpoint)
        self._generate_tags_func()
        self._apply_image_rating_filter(content_filter_level)

    def submit_answer(self, selected_tags_list: list, username=None) -> None:
        self._set_submitted_tags(selected_tags_list)

        logger.info(
            f'Correct answers: {len(self.correct_answers)}. '
            f'Incorrect answers: {len(self.incorrect_answers)}. '
            f'Net Points: {self.gained_points}'
        )

        if username:
            saved_path = self._save_current_image()
            question_id = self._store_question_data(file_path=saved_path)
            self._update_user_data(username, question_id)

    def _update_user_data(self, username: str, question_id: int):
        # Add the stored question to the attempted questions list of the user
        self.db_manager.add_attempted_question(username, question_id, self._submitted_tags)

        # Increment the user experience with self.gained_exp
        self.db_manager.increment_user_experience(username, self.gained_exp)

    def _save_current_image(self) -> str:    
        def uniquify(path: str) -> str:
            filename, extension = os.path.splitext(path)
            counter = 1
            while os.path.exists(path):
                path = filename + f" ({counter})" + extension
                counter += 1
            return path

        # Check if IMG_DIR exists, if not create it
        if not os.path.exists(IMG_DIR):
            os.makedirs(IMG_DIR)

        filename = f"img_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        file_path = os.path.join(IMG_DIR, filename)
        unique_file_path = uniquify(file_path)
        self.original_image.save(unique_file_path)
        logger.info(f'Image saved: {unique_file_path}')
        return unique_file_path

    def _store_question_data(self, file_path: str) -> int:
        """ Store the question data in the database. """
        question = self.db_manager.store_question(
            image_file_path=file_path,
            generation_info=self.image_gen_info,
            image_rating=self.image_rating,
            correct_tags=tags_to_json(self.image_tags),
            false_tags=tags_to_json(self.false_tags),
            generation_time=self.image_gen_time
        )
        return question.id

    def create_points_calculators(self) -> tuple:
        """
        Create functions to calculate points based on the given threshold.

        This method returns two calculators:
        1. `gained_points`: Calculates points gained for correct tags based on their weight.
        2. `lost_points`: Calculates points lost for incorrect tags based on their weight.

        For a general threshold:
            If weight = threshold: 
                `gained_points` returns 1
                `lost_points` returns 0
            If weight = 1.0:
                `gained_points` returns 1.25
                `lost_points` returns 0 (since it's a correct tag)
            If weight = 0.0:
                `gained_points` returns 0 (since it's below threshold and considered incorrect)
                `lost_points` returns -2

        Returns:
        - tuple: A tuple containing two functions:
            1. `gained_points` which takes a tag weight and returns the points gained.
            2. `lost_points` which takes a tag weight and returns the points lost (negative value).
        """
        threshold = self.wd14_tagger.general_tag_threshold

        # Ensure threshold is not 1.0 to prevent division by zero
        if threshold == 1.0:
            threshold -= 1e-9

        def gained_points(weight: float) -> float:
            if weight < threshold:
                return 0
            return 1 + 0.25 * (weight - threshold) / (1.0 - threshold)

        def lost_points(weight: float) -> float:
            if weight >= threshold:
                return 0
            return -2 * (threshold - weight) / threshold

        return gained_points, lost_points

    def get_tag_wiki(self, tag_name: str):
        return self.danbooru_api.get_tag_wiki(tag_name)

    @staticmethod
    def get_level_info(total_exp: int) -> (int, int, int):
        """
        Calculate the current level, EXP for that level, and the total EXP required to 
        reach the next level based on total experience points.

        :param total_exp: Total experience points.
        :return: (current_level, exp_for_current_level, exp_required_for_next_level)
        """
        
        level = 1
        exp_needed = 250

        while total_exp >= exp_needed:
            total_exp -= exp_needed
            level += 1
            exp_needed += 250

        exp_for_current_level = total_exp
        exp_required_for_next_level = exp_needed

        return level, exp_for_current_level, exp_required_for_next_level
    
    def submit_feedback(self, user_input: str, user_name: str, selected_tags: list = None):
        self.user_feedback.process_feedback(
            user_input, 
            self.image_gen_info, 
            user_name, 
            self.image_gen_time, 
            self.image_rating, 
            self.image_tags, 
            self.false_tags, 
            self.difficulty_level,
            selected_tags
        )
