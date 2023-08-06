import asyncio, aiohttp, random
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

from sd_api_wrapper import mock_generate_image as sd_generate_image, get_progress
from shared import SDModels, EventHandler, RANDOM_MODEL_OPT_STRING, ObserverContext
from wd14_tagging.wd14_tagging import WD14Tagger
from content_filter import ContentFilter


class App:
    def __init__(self):
        self._current_wd14_data = None
        self._current_image = None

        self.event_handler = EventHandler()
        self.wd14_tagger = WD14Tagger()
        self.content_filter = ContentFilter()

    def _get_sd_image(self, prompt: str, sd_model_opt: str):
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
            self.event_handler.notify_observers(0.0, f'Loading SD checkpoint {sd_model_opt}  ...')
            sd_model = SDModels.get_sd_model(sd_model_opt) if sd_model_opt != RANDOM_MODEL_OPT_STRING else random.choice(SDModels.get_checkpoints_names())

            async with aiohttp.ClientSession() as session:  # Run the async function and wait for it to complete
                post_task = asyncio.create_task(sd_generate_image(sd_model=sd_model, prompt=prompt))    # Create the POST task for the image request
                while not post_task.done():
                    progress, eta, current_step, total_steps, current_image = await get_progress(session)
                    self.event_handler.notify_observers(progress, f'Generating image - ETA: {eta:.2f}s - Step: {current_step}/{total_steps}')
                    await asyncio.sleep(1)

                image_with_metadata = await post_task   # Get the result of the POST task
                return image_with_metadata

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
    
    def _get_wd14_tags(self, img):
        with ObserverContext(self.wd14_tagger.event_handler, self.event_handler.get_latest_observer()):
            img, tags = self.wd14_tagger.tag_image(img)
        return img, tags
    
    def _set_current_wd14_data(self, img_wd14_data):
        self._current_wd14_data = img_wd14_data

    def _set_current_image(self, img: Image.Image):
        self._current_image = img

    def get_current_tags(self):
        return self._current_wd14_data['tags']
    
    def get_current_rating(self):
        return self._current_wd14_data['rating']
    
    def get_current_image(self):
        return self._current_image

    def generate_round(self, prompt: str, checkpoint: str):
        img = self._get_sd_image(prompt, checkpoint)

        # wd14_data = {'rating': 'general', 'tags': [('sky', 0.9), ('scenery', 0.88), ('star (sky)', 0.83), ('cloud', 0.81), ('starry sky', 0.76),
        #                  ('whale', 0.75), ('outdoors', 0.72), ('bird', 0.67), ('night', 0.47), ('night sky', 0.41), ('sunset', 0.4), ('animal', 0.39), ('shooting star', 0.37),        
        #                  ('cloudy sky', 0.35)]}

        img, wd14_data = self._get_wd14_tags(img)
        self._set_current_wd14_data(wd14_data)

        self.event_handler.notify_observers(0.9, 'Checking image rating against filters ...')

        if self.content_filter.is_rating_filtered(self.get_current_rating()):
            img = self.content_filter.get_blurred_image(self.get_current_image())
            print('oh no!') # TODO

        self._set_current_image(img)
