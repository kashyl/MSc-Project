import asyncio, aiohttp, random, re
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from math import floor

from custom_logging import logger
from sd_api_wrapper import generate_image as sd_generate_image, get_progress, mock_generate_image
from shared import SDModels, EventHandler, RANDOM_MODEL_OPT_STRING, ObserverContext, DifficultyLevels, DIFFICULTY_LEVEL_TAG_RATIO
from wd14_tagging.wd14_tagging import WD14Tagger
from content_filter import ContentFilter

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
        self._original_image = None # initial image, before post-process, content-filter blur etc
        self._image_gen_info = None
        self._difficulty_level = None

        self.event_handler = EventHandler()
        self.wd14_tagger = WD14Tagger()
        self.content_filter = ContentFilter()

        # Set function pointers based on the debug flags
        self._generate_image_func = self._generate_image if not debug_mock_image else self._mock_gen_image
        self._generate_tags_func = self._generate_tags if not debug_mock_tags else self._mock_gen_tags

    def _clear_round_data(self):
        self._image = None
        self._original_image = None
        self._image_gen_info = None
        self._difficulty_level = None

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

    def _set_original_image(self, img: Image.Image):
        self._original_image = img

    def _set_image_generation_info(self, sd_gen_info):
        self._image_gen_info = sd_gen_info

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
    def image_tags(self):
        return self.wd14_tagger.image_tags
    
    @property
    def image_rating(self):
        return self.wd14_tagger.image_rating
    
    @property
    def image_rating_and_tags(self):
        return self.wd14_tagger.image_rating_and_tags

    @property
    def false_tags(self):
        return self.wd14_tagger.random_false_tags

    @property
    def image(self):
        return self._image

    @property
    def tags_to_display(self):
        combined_tags = self.image_tags + self.false_tags
        random.shuffle(combined_tags)
        return combined_tags

    def _generate_image(self, prompt: str, checkpoint: str):
        generated_image = self._run_sd_generate(prompt, checkpoint)
        self._set_original_image(generated_image)
        self._set_image(generated_image)
        self._extract_image_generation_info()

    def _generate_tags(self):
        self._run_wd14_tagger()
        self._set_image(self.wd14_tagger.add_wd14_metadata_to_image(self.image))

        self.event_handler.notify_observers(0.9, 'Checking image rating against filters ...')
        if self.content_filter.is_rating_filtered(self.image_rating):
            self._set_image(self.content_filter.blur_image(self.image))
            print('oh no!') # TODO

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

    def _mock_gen_image(self, *args):
        self._set_image(mock_generate_image())    # get mock image for debugging
        self._set_image_generation_info(DEBUG_MOCK_GEN_INFO)

    def _mock_gen_tags(self, *args):
        self.wd14_tagger.mock_generate_tags()
        self.wd14_tagger.mock_gen_false_tags()

    def generate_round(self, prompt: str, checkpoint: str, difficulty: str):
        self._clear_round_data()
        self._set_difficulty_level(difficulty)
        self._generate_image_func(prompt, checkpoint)
        self._generate_tags_func()
