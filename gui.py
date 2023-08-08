import gradio as gr
from app import App
from shared import SDModels, RANDOM_MODEL_OPT_STRING, ObserverContext, DifficultyLevels, GUIFiltersLabels

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

    def generate_question(self, prompt: str, sd_model: str, content_filter_level: str, difficulty: str, gr_progress=gr.Progress()):
        with ObserverContext(self.app.event_handler, GUIProgressObserver(gr_progress)):
            self.app.generate_round(prompt, sd_model, content_filter_level, difficulty)

        img = self.app.image
        tags = self.app.tags_to_display
        gen_info = self.app.image_gen_info

        if self.app.image_is_filtered:
            gr.Warning(f'Image filtered due to content rating ({self.app.image_rating}).')

        return (
            self.ui_update_image(img),
            self.ui_update_tags(tags),
            self.ui_update_gen_info(gen_info),
            self.ui_update_reveal_btn_wrapper_visibility(),
            self.ui_update_filtered_disclaimer_lbl()
        )

    def ui_update_tags(self, image_tags):
        display_tags = not self.app.image_is_filtered
        try:
            return gr.CheckboxGroup.update(
                choices=[tag for tag, weight in image_tags], 
                visible=display_tags,
                interactive=True,
                info='Select the tags that correspond to the displayed image.'
            )
        except ValueError as e:
            raise ValueError(f'{e}\nimage_tags parameter value: {image_tags}')
        
    def ui_reveal_content(self):
        return self.ui_show_image_tags(), self.ui_reveal_image(), self.ui_update_reveal_btn_wrapper_visibility(False)

    def ui_reveal_image(self):
        original_image = self.app.original_image
        return gr.Image.update(value=original_image)

    def ui_show_image_tags(self):
        return gr.CheckboxGroup.update(visible=True)

    def ui_update_reveal_btn_wrapper_visibility(self, override_display=None):
        display_btn = override_display if override_display is not None else self.app.image_is_filtered
        return gr.Box.update(visible=display_btn)

    def ui_update_filtered_disclaimer_lbl(self):
        if self.app.image_is_filtered:
            return gr.Markdown.update(value=
                f"""
                # Image Filtered Based on Content Rating
                Due to the unpredictable nature of AI, there might be instances where the generated content is inappropriate. 
                The generated image received a rating of **{self.app.image_rating}** and has been filtered in accordance with your settings.

                You have the options to:
                * Generate a new image.
                * Reveal the filtered content.
                * Adjust content rating preferences in settings.
                """)
        else:
            return gr.Markdown.update(value='')

    def ui_show_tags_wrapper(self):
        return gr.Column.update(visible=True)

    def ui_update_image(self, image):
        return gr.Image.update(value=image, label='AI Generated Image', show_share_button=True)
    
    def ui_update_gen_info(self, img_sd_gen_info):
        return gr.Markdown.update(value=img_sd_gen_info)
    
    def ui_update_gen_info_wrapper(self, gen_info):
        return gr.Accordion.update(visible=bool(gen_info))

    def launch(self):
        with gr.Blocks(css="footer {visibility: hidden}") as demo:
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
                            generated_image = gr.Image(elem_id='generated-image', label='Click the button to generate an image.')
                            with gr.Row():
                                with gr.Accordion('Generation Details', visible=False, open=False) as gen_info_wrapper:
                                    gen_info = gr.Markdown()
                        with gr.Column():
                            with gr.Box(visible=False) as reveal_content_wrapper:
                                with gr.Column():
                                    filter_disclaimer_lbl = gr.Markdown()
                                    reveal_content_btn = gr.Button('Reveal Content')
                            image_tags = gr.CheckboxGroup(
                                choices=None, 
                                label="Image Tags", 
                                info="Tags will appear below once the image is generated.", 
                                interactive=True, 
                                visible=True
                            )

            with gr.Tab(label='Settings'):
                sd_checkpoint = gr.Dropdown(
                    choices=[
                        RANDOM_MODEL_OPT_STRING,
                        *SDModels.get_gui_model_names_list()
                    ],
                    label='Stable Diffusion Model',
                    info='Select the pre-trained model checkpoint (.safetensors/.ckpt) to use for image generation.',
                    show_label=True,
                    value=RANDOM_MODEL_OPT_STRING,
                    interactive=True
                )
                content_filter_rating = gr.Radio(
                    [enum.value for enum in GUIFiltersLabels],
                    value=GUIFiltersLabels.GENERAL.value, 
                    label="Content Rating Filter", 
                    info = ("Select a content rating; any generated images rated above the chosen level will be blurred out.")
                )

            # Events
            generate_btn.click(
                self.generate_question, 
                inputs=[
                    custom_prompt, 
                    sd_checkpoint,
                    content_filter_rating, 
                    game_difficulty
                ], 
                outputs=[
                    generated_image, 
                    image_tags, 
                    gen_info,
                    reveal_content_wrapper,
                    filter_disclaimer_lbl
                ]
            )
            gen_info.change(fn=self.ui_update_gen_info_wrapper, inputs=gen_info, outputs=gen_info_wrapper)
            reveal_content_btn.click(
                fn=self.ui_reveal_content,
                outputs=[
                    image_tags,
                    generated_image,
                    reveal_content_wrapper
                ]
            )

        demo.queue(concurrency_count=20).launch()
