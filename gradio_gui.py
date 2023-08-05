import gradio as gr
import numpy as np
from sd_api_wrapper import mock_generate_image as sd_generate_image, get_progress, save_image
from image_tagging import ImageTagger
from config import SDModels
from concurrent.futures import ThreadPoolExecutor
import asyncio, aiohttp, time

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
        pass

    def generate_image_gradio(self, prompt: str, checkpoint: str, gr_progress: gr.Progress):
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
            update_progress(gr_progress, 0.0, f'Loading SD checkpoint {checkpoint}  ...')
            sd_model = SDModels.get_sd_model(checkpoint)

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
    
    def generate_and_tag(self, prompt: str, checkpoint: str, gr_progress=gr.Progress()):
        img = self.generate_image_gradio(prompt, checkpoint, gr_progress)
        img, tags = self.tag_using_wd14(img, gr_progress)

        update_progress(gr_progress, 0.9, 'Checking image rating against filters ...')
        tags['rating'] = 'explicit' # DEBUG
        # if tags['rating'] in FILTERED_RATINGS:
        #     print('oh no!') # TODO
        #     from PIL import Image, ImageFilter
        #     img = img.filter(ImageFilter.GaussianBlur(radius=50))
        return img

    def launch(self):
        with gr.Blocks(css="footer {visibility: hidden}") as demo:
            gr.Markdown("Generate image.")
            with gr.Accordion("Settings", open=False):
                sd_checkpoint = gr.Dropdown(
                    choices=[
                        SDModels.CheckpointNamesGUI.GHOSTMIX,
                        SDModels.CheckpointNamesGUI.REALISTICVISION,
                        SDModels.CheckpointNamesGUI.AOM3A1B,
                        SDModels.CheckpointNamesGUI.ANYTHINGV3,
                        SDModels.CheckpointNamesGUI.SCHAUXIER,
                        SDModels.CheckpointNamesGUI.KANPIRO,
                        SDModels.CheckpointNamesGUI.DREAMSHAPER,
                        SDModels.CheckpointNamesGUI.REVANIMATED,
                        SDModels.CheckpointNamesGUI.MISTOON
                    ],
                    label='Stable Diffusion Checkpoint (Model)',
                    show_label=True,
                    value=SDModels.CheckpointNamesGUI.GHOSTMIX,
                    interactive=True
                )

            with gr.Row():
                with gr.Box():
                    generated_image = gr.Image(label='AI Generated Image', elem_id='generated-image', show_share_button=True)
                text_input = gr.Textbox('whale on deep blue sky, white birds, star, thick clouds')
            generate_btn = gr.Button("Generate")
            generate_btn.click(self.generate_and_tag, inputs=[text_input, sd_checkpoint], outputs=generated_image)

        demo.queue(concurrency_count=20).launch()

grui = GradioUI()
grui.launch()
