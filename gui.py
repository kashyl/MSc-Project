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
            self.ui_update_filtered_disclaimer_lbl(),
            self.ui_update_generate_btn_display(False),
            self.ui_update_submit_btn_display(),
            self.ui_show_results_wrapper(False)
        )
    
    def ui_update_tags(self, image_tags):
        display_tags = not self.app.image_is_filtered
        try:
            return gr.CheckboxGroup.update(
                choices=[tag for tag, weight in image_tags], 
                visible=display_tags,
                interactive=True,
                info='Select the tags that best match the displayed image.'
            )
        except ValueError as e:
            raise ValueError(f'{e}\nimage_tags parameter value: {image_tags}')
        
    def ui_reveal_content(self):
        return (
            self.ui_show_image_tags(), 
            self.ui_reveal_image(), 
            self.ui_update_reveal_btn_wrapper_visibility(False),
            self.ui_update_submit_btn_display(True)
        )

    def ui_reveal_image(self):
        original_image = self.app.original_image
        return gr.Image.update(value=original_image)

    def ui_show_image_tags(self, display=True):
        return gr.CheckboxGroup.update(visible=display)

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

    def ui_update_generate_btn_display(self, override_display=None):
        display_btn = override_display if override_display is not None else not self.app.image_is_filtered
        return gr.Button.update(visible=display_btn)

    def ui_update_submit_btn_display(self, override_display=None):
        display_btn = override_display if override_display is not None else not self.app.image_is_filtered
        return gr.Button.update(visible=display_btn)

    def on_submit(self, selected_tags: list):
        self.app.submit_selected_tags(selected_tags)
        return (
            self.ui_update_submit_btn_display(False),
            self.ui_show_image_tags(False),
            self.ui_show_results_wrapper(),
            self.ui_update_results_markdown(),
            self.ui_update_tag_wiki_search()
        ) 

    def ui_show_results_wrapper(self, display=True):
        return gr.Box.update(visible=display)

    def ui_update_results_markdown(self):
        exp = self.app.gained_exp
        correct, incorrect, missed = (
            self.app.tag_names_only(attr) 
            for attr in (
                self.app.correct_answers, 
                self.app.incorrect_answers, 
                self.app.missed_answers
            )
        )

        # Calculate accuracy
        total_tags = len(correct) + len(incorrect) + len(missed)
        accuracy = len(correct) / total_tags if total_tags != 0 else 0
        
        # Decide on the title based on accuracy
        title = "# Well Done!" if accuracy > 0.5 else "# Keep Going"
        
        # Construct identification line with/without exclamation based on accuracy
        identification_line = f"You identified **{len(correct)}** out of **{total_tags}** tags correctly"
        identification_line += "!" if accuracy > 0.5 else "."
        
        lines = [title, identification_line]

        # Only add EXP gain message if EXP > 0
        if exp > 0:
            lines.append(f"+{exp} XP")
            gr.Info(f"You've earned {exp} XP!")
        
        # Construct individual lines
        lines.extend([
            f"🌟 **Correct answers**: {', '.join(correct)}",
            f"🤔 **Not Quite**: {', '.join(incorrect)}",
            f"🔍 **You Missed These**: {', '.join(missed)}"
        ])
        
        # Combine and return
        markdown_result = "\n\n".join(lines)
        return gr.Markdown.update(value=markdown_result)

    def ui_update_tag_wiki_search(self):
        if not self.app.image_tags and not self.app.false_tags:
            return gr.Dropdown.update(choices=[], visible=False, interactive=False)
        else:
            return gr.Dropdown.update(
                choices=[
                    tag 
                    for tags_sublist in [
                        self.app.tag_names_only(self.app.correct_answers), 
                        self.app.tag_names_only(self.app.incorrect_answers), 
                        self.app.tag_names_only(self.app.missed_answers)
                    ] 
                    for tag in tags_sublist
                ],
                visible=True,
                interactive=True
            )

    def ui_update_tag_wiki_result(self, tag_name: str):
        tag_info = 'do api module'

        if not tag_info:
            return gr.Markdown.update(value=None, visible=False)
            
        return gr.Markdown.update(value=tag_info, visible=True)

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
                                    with gr.Row():
                                        on_filtered_generate_new_btn = gr.Button('Generate New Image')
                                        reveal_content_btn = gr.Button('Reveal Content')

                            image_tags = gr.CheckboxGroup(
                                choices=None, 
                                label="Image Tags", 
                                info="Tags will appear below once the image is generated.", 
                                interactive=True, 
                                visible=True
                            )
                            generate_btn = gr.Button("Generate New Image")
                            submit_btn = gr.Button("Submit", visible=False)

                            with gr.Box(visible=False) as results_wrapper:
                                with gr.Column():
                                    results_md = gr.Markdown()
                                    results_tag_wiki_search_dropdown = gr.Dropdown(
                                        allow_custom_value=False, 
                                        multiselect=False,
                                        label="Tag Details Lookup",
                                        info="Select a tag to retrieve its detailed description from the tag wiki.")
                                    results_tag_wiki_result_md = gr.Markdown()
                                    results_generate_new_btn = gr.Button("Generate New Image")

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
            def bind_generate_click_event(button: gr.Button):
                button.click(
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
                        filter_disclaimer_lbl,
                        generate_btn,
                        submit_btn,
                        results_wrapper
                    ]
                )

            bind_generate_click_event(generate_btn)
            bind_generate_click_event(on_filtered_generate_new_btn)
            bind_generate_click_event(results_generate_new_btn)

            gen_info.change(fn=self.ui_update_gen_info_wrapper, inputs=gen_info, outputs=gen_info_wrapper)

            reveal_content_btn.click(
                fn=self.ui_reveal_content,
                outputs=[
                    image_tags,
                    generated_image,
                    reveal_content_wrapper,
                    submit_btn
                ]
            )

            submit_btn.click(
                fn=self.on_submit, 
                inputs=image_tags, 
                outputs=[
                    submit_btn,
                    image_tags,
                    results_wrapper,
                    results_md,
                    results_tag_wiki_search_dropdown
                ]
            )

            results_tag_wiki_search_dropdown.change(
                fn=self.ui_update_tag_wiki_result,
                inputs=results_tag_wiki_search_dropdown,
                outputs=results_tag_wiki_result_md
            )

        demo.queue(concurrency_count=20).launch()
