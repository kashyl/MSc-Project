import numpy as np
import gradio as gr


def generate_image(x):
    return np.fliplr(x)


with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("Flip text or image files using this demo.")
    with gr.Row():
        text_input = gr.Textbox()
        image_output = gr.Image()
    image_button = gr.Button("Generate")

    with gr.Accordion("Open for More!"):
        gr.Markdown("Look at me...")

    image_button.click(generate_image, inputs=text_input, outputs=image_output)

demo.launch()
