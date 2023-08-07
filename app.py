import asyncio, aiohttp, random
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

from sd_api_wrapper import mock_generate_image as sd_generate_image, get_progress
from shared import SDModels, EventHandler, RANDOM_MODEL_OPT_STRING, ObserverContext
from wd14_tagging.wd14_tagging import WD14Tagger
from content_filter import ContentFilter


class App:
    def __init__(self):
        self._image = None

        self.event_handler = EventHandler()
        self.wd14_tagger = WD14Tagger()
        self.content_filter = ContentFilter()

    def _generate_image(self, prompt: str, sd_model_opt: str):
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
    
    def _generate_tags(self):
        with ObserverContext(self.wd14_tagger.event_handler, self.event_handler.get_latest_observer()):
            self.wd14_tagger.generate_tags(self.image)

    def _set_image(self, img: Image.Image):
        self._image = img

    def _set_image_false_tags(self, tags):
        self._image_false_tags = tags
    
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

    def generate_round(self, prompt: str, checkpoint: str):
        self._set_image(self._generate_image(prompt, checkpoint))

        self._generate_tags()
        self._set_image(self.wd14_tagger.add_wd14_metadata_to_image(self.image))

        self.event_handler.notify_observers(0.9, 'Checking image rating against filters ...')
        if self.content_filter.is_rating_filtered(self.image_rating):
            self._set_image(self.content_filter.get_blurred_image(self.image))
            print('oh no!') # TODO

        self.wd14_tagger.generate_random_false_tags(5)  # TODO: count
