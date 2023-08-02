import gradio as gr
from sd_api_wrapper import generate_image as sd_generate_image
from config import SDModels
import asyncio

def generate_image_gradio(x):
    sd_model = SDModels.sardonyxREDUX_v20
    prompt = 'whale on deep blue sky, white birds, star, thick clouds'

    # Run the async function and wait for it to complete
    image_with_metadata = asyncio.run(sd_generate_image(sd_model=sd_model, prompt=prompt))

    return image_with_metadata

with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("Generate image.")
    with gr.Row():
        text_input = gr.Textbox()
        image_output = gr.Image()
    image_button = gr.Button("Generate")

    with gr.Accordion("Open for More!"):
        gr.Markdown("Look at me...")

    image_button.click(generate_image_gradio, inputs=text_input, outputs=image_output)

demo.launch()
