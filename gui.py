import gradio as gr
from app import App
from shared import SDModels, RANDOM_MODEL_OPT_STRING, ObserverContext, DifficultyLevels

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

    def generate_question(self, prompt: str, sd_model: str, difficulty: str, gr_progress=gr.Progress()):
        with ObserverContext(self.app.event_handler, GUIProgressObserver(gr_progress)):
            self.app.generate_round(prompt, sd_model, difficulty)

        img = self.app.image
        tags = self.app.tags_to_display
        gen_info = self.app.image_gen_info

        return self.ui_update_image(img), self.ui_update_tags(tags), self.ui_update_gen_info(gen_info)

    def ui_update_tags(self, image_tags):
        try:
            return gr.CheckboxGroup.update(
                choices=[tag for tag, weight in image_tags], 
                visible=True,
                interactive=True
            )
        except ValueError as e:
            raise ValueError(f'{e}\nimage_tags parameter value: {image_tags}')
        
    def ui_update_image(self, image):
        return gr.Image.update(value=image, label='AI Generated Image', show_share_button=True)
    
    def ui_update_gen_info(self, img_sd_gen_info):
        return gr.Markdown.update(value=img_sd_gen_info)
    
    def ui_update_gen_info_wrapper(self, gen_info):
        return gr.Accordion.update(visible=bool(gen_info))

    def launch(self):
        with gr.Blocks(css="footer {visibility: hidden}") as demo:
            gr.Markdown("Generate image.")

            with gr.Tab('Main'):
                with gr.Row():
                    custom_prompt = gr.Textbox('whale, deep blue sky, white birds, star, thick clouds', label='Custom prompt')
                    game_difficulty = gr.Radio(
                        [d.value for d in DifficultyLevels],
                        value=DifficultyLevels.NORMAL.value, 
                        label="Difficulty Level", 
                        info = ("Difficulty influences the true/false tag ratio, time limit, and gained EXP rewards.")
                    )

                with gr.Row():
                    generate_btn = gr.Button("Generate Question")

                with gr.Box():
                    with gr.Row():
                        with gr.Column():
                            generated_image = gr.Image(elem_id='generated-image', label='Click the button to generate an image!')
                            with gr.Accordion('Image Generation Info', visible=False, open=False) as gen_info_wrapper:
                                gen_info = gr.Markdown(value='a')
                        image_tags = gr.CheckboxGroup(choices=None, label="Tags", info="info message", interactive=True, visible=False)

            with gr.Tab(label='Settings'):
                with gr.Row():
                    sd_checkpoint = gr.Dropdown(
                        choices=[
                            RANDOM_MODEL_OPT_STRING,
                            *SDModels.get_gui_model_names_list()
                        ],
                        label='Stable Diffusion Checkpoint (Model)',
                        show_label=True,
                        value=RANDOM_MODEL_OPT_STRING,
                        interactive=True
                    )

            # Events
            generate_btn.click(
                self.generate_question, 
                inputs=[
                    custom_prompt, 
                    sd_checkpoint, 
                    game_difficulty
                ], 
                outputs=[
                    generated_image, 
                    image_tags, 
                    gen_info
                ]
            )
            gen_info.change(fn=self.ui_update_gen_info_wrapper, inputs=gen_info, outputs=gen_info_wrapper)

        demo.queue(concurrency_count=20).launch()
