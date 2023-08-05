import gradio as gr
import numpy as np
from sd_api_wrapper import mock_generate_image as sd_generate_image, get_progress, save_image
from image_tagging import ImageTagger
from config import SDModels, CheckpointNamesGUI
from concurrent.futures import ThreadPoolExecutor
import asyncio, aiohttp, time
import random

RANDOM_MODEL_OPT = 'Select a model at random for each generation'

DEFAULT_FILTER_LEVEL = 1
image_tagger = ImageTagger(content_filter_level=DEFAULT_FILTER_LEVEL)

def update_progress(progress_component: gr.Progress, prog_val: str, prog_msg: str):
    """
    Updates a Gradio progress component with the given progress value and message.

    :param progress_component: A Gradio progress component that will be updated.
    :type progress_component: gr.Progress
    :param prog_val: The progress value to be set, typically in the range [0, 1].
    :type prog_val: float
    :param prog_msg: A descriptive message to accompany the progress value, e.g., a status update or ETA.
    :type prog_msg: str
    """
    progress_component(prog_val, desc=prog_msg)
    #time.sleep(0.01) # Manual Gradio progress bar may need a delay to work?

class GUIProgressObserver:
    def __init__(self, gr_progress):
        self.gr_progress = gr_progress
    def update(self, progress_val, progress_msg):
        update_progress(self.gr_progress, progress_val, f'{progress_msg}')


class GradioUI:
    def __init__(self):
        self.current_img_tags = None
        self.current_img_rating = None

    def ui_tags_update(self):
        return gr.CheckboxGroup.update(choices=self.current_img_tags, visible=True)

    def generate_image_gradio(self, prompt: str, sd_model_opt: str, gr_progress: gr.Progress):
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
            update_progress(gr_progress, 0.0, f'Loading SD checkpoint {sd_model_opt}  ...')
            sd_model = SDModels.get_sd_model(sd_model_opt) if sd_model_opt != RANDOM_MODEL_OPT else random.choice(SDModels.get_checkpoints_names())

            async with aiohttp.ClientSession() as session:  # Run the async function and wait for it to complete
                post_task = asyncio.create_task(sd_generate_image(sd_model=sd_model, prompt=prompt))    # Create the POST task for the image request
                while not post_task.done():
                    progress, eta, current_step, total_steps, current_image = await get_progress(session)
                    update_progress(gr_progress, progress, f'Generating image - ETA: {eta:.2f}s - Step: {current_step}/{total_steps}')
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
    
    def tag_using_wd14(self, img, gr_progress):
        ui_progress_observer = GUIProgressObserver(gr_progress)
        image_tagger.event_handler.add_observer(ui_progress_observer)
        img, tags = image_tagger.tag_image(img)
        image_tagger.event_handler.remove_observer(ui_progress_observer)
        return img, tags
    
    def generate_and_tag_image(self, prompt: str, checkpoint: str, gr_progress=gr.Progress()):
        img = self.generate_image_gradio(prompt, checkpoint, gr_progress)

        # img, wd14_data = self.tag_using_wd14(img, gr_progress) DEBUG TODO
        wd14_data = {'rating': 'general', 'tags': [('sky', 0.9), ('scenery', 0.88), ('star (sky)', 0.83), ('cloud', 0.81), ('starry sky', 0.76),
                         ('whale', 0.75), ('outdoors', 0.72), ('bird', 0.67), ('night', 0.47), ('night sky', 0.41), ('sunset', 0.4), ('animal', 0.39), ('shooting star', 0.37),        
                         ('cloudy sky', 0.35)]}

        self.current_img_tags = wd14_data['tags']
        self.current_img_rating = wd14_data['rating']

        update_progress(gr_progress, 0.9, 'Checking image rating against filters ...')
        if wd14_data['rating'] == 'filtered':
            print('oh no') # TODO: probably don't generate the game and notify user image was filtered, try again.

        return img, self.ui_tags_update()

    def launch(self):
        with gr.Blocks(css="footer {visibility: hidden}") as demo:
            gr.Markdown("Generate image.")

            with gr.Tab('Main'):
                custom_prompt = gr.Textbox('whale on deep blue sky, white birds, star, thick clouds', label='Custom prompt')
                with gr.Row():
                    with gr.Box():
                        generated_image = gr.Image(label='AI Generated Image', elem_id='generated-image', show_share_button=True)
                    tags_list = gr.CheckboxGroup(choices=None, label="Tags", info="info message", interactive=True, visible=False)
                generate_btn = gr.Button("Generate")

            with gr.Tab(label='Settings'):
                with gr.Row():
                    sd_checkpoint = gr.Dropdown(
                        choices=[
                            RANDOM_MODEL_OPT,
                            *SDModels.get_checkpoints_names()
                        ],
                        label='Stable Diffusion Checkpoint (Model)',
                        show_label=True,
                        value=RANDOM_MODEL_OPT,
                        interactive=True
                    )

                # Events
                generate_btn.click(self.generate_and_tag_image, inputs=[custom_prompt, sd_checkpoint], outputs=[generated_image, tags_list])
                # generated_image.change(fn=self.ui_tags_update, outputs=tags_list)

        demo.queue(concurrency_count=20).launch()

grui = GradioUI()
grui.launch()
