import gradio as gr
import numpy as np
from sd_api_wrapper import generate_image as sd_generate_image, get_progress
from config import SDModels
import asyncio, aiohttp, time

async def generate_image_gradio(x):
    sd_model = SDModels.sardonyxREDUX_v20
    prompt = 'whale on deep blue sky, white birds, star, thick clouds'

    # Run the async function and wait for it to complete
    async with aiohttp.ClientSession() as session:
        # Create the POST task for the image request
        task_post = asyncio.create_task(sd_generate_image(sd_model=sd_model, prompt=prompt))

        # Check the progress and sleep every second while waiting for the POST task to complete
        while not task_post.done():
            preview = await get_progress(session)
            if preview:
                yield preview
            await asyncio.sleep(1)

    # Get the result of the POST task
    image_with_metadata = await task_post   # resp = {'images': ['bjkahsdjsa...', 'askjdhas...', 'ajkjhdashda...']}
    yield image_with_metadata

with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("Generate image.")
    with gr.Row():
        text_input = gr.Textbox()
        with gr.Box():
            image_output = gr.Image(label='Generated image', elem_id='generated-image', show_share_button=True)
            progress_bar = gr.Markdown()
    image_button = gr.Button("Generate")

    with gr.Accordion("Open for More!"):
        gr.Markdown("Look at me...")
        clean_imgs_btn = gr.Button("Clean Images")

        def progress_test(progress=gr.Progress()):
            progress_bar.update(visible=True)
            progress(0.2, desc="Collecting Images")
            time.sleep(1)
            progress(0.5, desc="Cleaning Images")
            time.sleep(1.5)
            progress(0.8, desc="Sending Images")
            time.sleep(1.5)
            progress_bar.update(visible=False)
        clean_imgs_btn.click(progress_test, outputs=progress_bar)

    image_button.click(generate_image_gradio, inputs=text_input, outputs=image_output)

demo.queue(concurrency_count=20).launch()
