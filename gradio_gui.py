import gradio as gr
import numpy as np
from sd_api_wrapper import generate_image as sd_generate_image, get_progress
from config import SDModels
import asyncio, aiohttp, time

class GradioUI:
    def __init__(self):
        pass

    async def generate_image_gradio(self, gr_progress=gr.Progress()):
        sd_model = SDModels.sardonyxREDUX_v20
        prompt = 'whale on deep blue sky, white birds, star, thick clouds'

        # Run the async function and wait for it to complete
        async with aiohttp.ClientSession() as session:
            # Create the POST task for the image request
            post_task = asyncio.create_task(sd_generate_image(sd_model=sd_model, prompt=prompt))
            while not post_task.done():    
                progress, eta, current_step, total_steps, current_image = await get_progress(session)
                gr_progress(progress, desc=f'Generating image - ETA: {eta:.2f}s - Step: {current_step}/{total_steps}')
                await asyncio.sleep(1)
                
        # Get the result of the POST task
        image_with_metadata = await post_task
        return image_with_metadata

    def launch(self):
        with gr.Blocks(css="footer {visibility: hidden}") as demo:
            gr.Markdown("Generate image.")
            with gr.Row():
                with gr.Box():
                    generated_image = gr.Image(label='Generated image', elem_id='generated-image', show_share_button=True)
                text_input = gr.Textbox()
            image_button = gr.Button("Generate")

            with gr.Accordion("Open for More!"):
                gr.Markdown("Look at me...")

            image_button.click(self.generate_image_gradio, outputs=generated_image)

        demo.queue(concurrency_count=20).launch()

grui = GradioUI()
grui.launch()
