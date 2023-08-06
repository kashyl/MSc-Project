import gradio as gr
from app import App
from shared import SDModels, RANDOM_MODEL_OPT_STRING, ObserverContext

class GUIProgressObserver:
    def __init__(self, gr_progress):
        self.gr_progress = gr_progress
    def update(self, progress_val, progress_msg):
        self.gr_progress(progress_val, f'{progress_msg}')

class GradioUI:
    def __init__(self, app: App):
        self.current_img_tags_and_weights = None
        self.current_img_rating = None
        self.app = app

    def generate_question(self, prompt: str, sd_model: str, gr_progress=gr.Progress()):
        with ObserverContext(self.app.event_handler, GUIProgressObserver(gr_progress)):
            self.app.generate_round(prompt, sd_model)

        img = self.app.get_current_image()
        tags = self.app.get_current_tags()

        return img, self.ui_update_tags(tags)

    def ui_update_tags(self, image_tags):
        return gr.CheckboxGroup.update(
            choices=[tag for tag, weight in image_tags], 
            visible=True,
            interactive=True
        )

    def launch(self):
        with gr.Blocks(css="footer {visibility: hidden}") as demo:
            gr.Markdown("Generate image.")

            with gr.Tab('Main'):
                with gr.Row():
                    custom_prompt = gr.Textbox('whale on deep blue sky, white birds, star, thick clouds', label='Custom prompt')
                    generate_btn = gr.Button("Generate Question")
                with gr.Row():
                    with gr.Box():
                        generated_image = gr.Image(label='AI Generated Image', elem_id='generated-image', show_share_button=True)
                    image_tags = gr.CheckboxGroup(choices=None, label="Tags", info="info message", interactive=True, visible=False)

            with gr.Tab(label='Settings'):
                with gr.Row():
                    sd_checkpoint = gr.Dropdown(
                        choices=[
                            RANDOM_MODEL_OPT_STRING,
                            *SDModels.get_checkpoints_names()
                        ],
                        label='Stable Diffusion Checkpoint (Model)',
                        show_label=True,
                        value=RANDOM_MODEL_OPT_STRING,
                        interactive=True
                    )

            # Events
            generate_btn.click(self.generate_question, inputs=[custom_prompt, sd_checkpoint], outputs=[generated_image, image_tags])
            #generated_image.change(fn=self.ui_display_new_image_tags, outputs=image_tags)

        demo.queue(concurrency_count=20).launch()
