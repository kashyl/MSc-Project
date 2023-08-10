import re
import gradio as gr
from app import App
from shared import SDModels, RANDOM_MODEL_OPT_STRING, ObserverContext, DifficultyLevels, GUIFiltersLabels, DIFFICULTY_LEVEL_EXP_GAIN
from db_manager import DEFAULT_STATE, GUIAlertType, DBResponse, UserFields

DEFAULT_GUEST_WELCOME = ('**👤 Playing as a Guest.** Log in or **sign up** to gain access to **play statistics**, '
                                '**question history**, and **compete on leaderboards**!'
)

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
        return gr.CheckboxGroup.update(
            choices=[tag for tag, weight in image_tags], 
            visible=display_tags,
            interactive=True,
            info='Select the tags that best match the displayed image.'
        )
        
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
            self.ui_update_tag_wiki_search(),
            self.ui_clear_tag_wiki_result(),
            self.ui_show_tag_wiki_result_wrapper(False),
            self.ui_update_generate_btn_display(True)
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
        
        # Construct individual lines
        lines.extend([
            f"🌟 **Correct answers**: {', '.join(correct)}",
            f"🤔 **Not Quite**: {', '.join(incorrect)}",
            f"🔍 **You Missed These**: {', '.join(missed)}"
        ])

        # Only add EXP gain message if EXP > 0
        if exp > 0:
            difficulty_enum = DifficultyLevels[self.app.difficulty_level.upper()]
            exp_gain_multiplier = DIFFICULTY_LEVEL_EXP_GAIN[difficulty_enum]
            lines.append(f'<span style="color:green">+{exp} XP</span> <span style="color:grey">(x{exp_gain_multiplier} {self.app.difficulty_level} difficulty multiplier)</span>')
            gr.Info(f"You've earned {exp} XP!")
        
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
        tag_title, tag_info = self.app.get_tag_wiki(tag_name)

        if not tag_info:
            no_info_msg = f"Couldn't retrieve information for the tag **{tag_name}**."
            return gr.Markdown.update(value=no_info_msg, visible=True)
            
         # Prepend the tag_title to the tag_info
        full_info = f"## {tag_title}\n\n{tag_info}"

        return (
            gr.Markdown.update(value=full_info, visible=True),
            self.ui_show_tag_wiki_result_wrapper(True)
        )

    def ui_clear_tag_wiki_result(self):
        return gr.Markdown.update(value=None)

    def ui_show_tag_wiki_result_wrapper(self, display=True):
        return gr.Box.update(visible=display)

    def ui_update_image(self, image):
        return gr.Image.update(value=image, label='AI Generated Image', show_share_button=True)
    
    def ui_update_gen_info(self, img_sd_gen_info):
        return gr.Markdown.update(value=img_sd_gen_info)
    
    def ui_update_gen_info_wrapper(self, gen_info):
        return gr.Accordion.update(visible=bool(gen_info))
    
    def ui_account_login_or_register_display(self, choice):
        if choice == "Login":
            return gr.Button.update(visible=True), gr.Button.update(visible=False)
        elif choice == "Register":
            return gr.Button.update(visible=False), gr.Button.update(visible=True)

    def ui_display_column(self, display=True):
        return gr.Column.update(visible=display)

    @staticmethod
    def _account_register_validate_input(username, password):
        """
        Username Requirements:

        Only letters and numbers.
        Minimum of 3 characters.
        Start with 2 letters.
        Maximum of 10 characters.
        Regex: ^[a-zA-Z]{2}[a-zA-Z0-9]{1,8}$

        Password Requirements:

        At least 4 characters.
        Maximum of 30 characters.
        For this, a simple length check will suffice.
        """
        username_pattern = r"^[a-zA-Z]{2}[a-zA-Z0-9]{1,8}$"
        if not re.match(username_pattern, username):
            gr.Warning("Username must start with 2 letters, be between 3 to 10 characters long, and contain only letters or numbers.")
            return False

        if len(password) < 4 or len(password) > 30:
            gr.Warning("Password must be between 4 to 30 characters long.")
            return False

        return True
    
    @staticmethod
    def _account_login_validate_input(username, password):
        """
        Username Requirements:
        Not empty.

        Password Requirements:
        Not empty.
        """

        if not username.strip():
            gr.Warning("Username cannot be empty.")
            return False

        if not password.strip():
            gr.Warning("Password cannot be empty.")
            return False

        return True

    @staticmethod
    def _handle_db_resp_message(response: DBResponse):
        if not response or not response.message_type or not response.message:
            return
        if response.message_type == GUIAlertType.INFO:
            gr.Info(response.message)
        elif response.message_type == GUIAlertType.WARNING:
            gr.Warning(response.message)
        elif response.message_type == GUIAlertType.ERROR:
            raise gr.Error(response.message)

    def account_register(self, username: str, password: str, state:dict):
        response = None
        if self._account_register_validate_input(username, password):
            response = self.app.db_manager.register(username, password, state)
        return self._finalize_login_response(response)

    def account_login(self, username: str, password: str, state: dict):
        response = None
        if self._account_login_validate_input(username, password):
            response = self.app.db_manager.login(username, password, state)
        return self._finalize_login_response(response)

    def _finalize_login_response(self, response):
        self._handle_db_resp_message(response)
        if response is None or response.state[UserFields.NAME] is None:
            return (
                DEFAULT_STATE,
                self.ui_display_column(True),
                self.ui_display_column(False),
                *self.ui_clear_main_tab_user_info()
            )
        return (
            response.state, 
            self.ui_display_column(False),  # account login/register form
            self.ui_display_column(True),   # account details wrapper
            *self.ui_update_main_tab_user_info(response.state)
        )

    def ui_update_main_tab_user_info(self, state):
        # Extract the values from the state
        username = state[UserFields.NAME]
        level, exp_for_current_level, exp_required_for_next_level = self.app.get_level_info(state[UserFields.EXP])
        accuracy = state[UserFields.ACCURACY]
        ans_count = state[UserFields.ANS_COUNT]

        # Create the markdown strings
        user_md = f'👤 Hi there, **{username}**!'
        exp_md = f"**🏅 Level:** {level} | **✨ XP:** {exp_for_current_level}/{exp_required_for_next_level}"
        accuracy_md = f"**🎯 Accuracy rating:** {accuracy}%"
        ans_md = f"**❓ Questions answered:** {ans_count}"

        return (
            gr.Markdown.update(value=user_md),
            gr.Markdown.update(value=exp_md, visible=True),
            gr.Markdown.update(value=accuracy_md, visible=True),
            gr.Markdown.update(value=ans_md, visible=True)
        )
        
    def ui_clear_main_tab_user_info(self):
        return (
            gr.Markdown.update(value=DEFAULT_GUEST_WELCOME),
            gr.Markdown.update(value=None, visible=False),
            gr.Markdown.update(value=None, visible=False),
            gr.Markdown.update(value=None, visible=False)
        )

    def launch(self):
        with gr.Blocks(css="footer {visibility: hidden}") as demo:
            gr_state = gr.State(value=DEFAULT_STATE)

            with gr.Tab('Main'):
                with gr.Box():
                    with gr.Row():
                        main_tab_user_name_md = gr.Markdown(value=(DEFAULT_GUEST_WELCOME))
                        main_tab_user_exp_md = gr.Markdown(visible=False)
                        main_tab_user_accuracy_md = gr.Markdown(visible=False)
                        main_tab_user_questions_count_md = gr.Markdown(visible=False)
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

                            submit_btn = gr.Button("Submit", visible=False)

                            with gr.Box(visible=False) as results_wrapper:
                                with gr.Column():
                                    results_md = gr.Markdown()
                                    results_tag_wiki_search_dropdown = gr.Dropdown(
                                        allow_custom_value=False, 
                                        multiselect=False,
                                        label="Tag Details Lookup",
                                        info="Select a tag to retrieve its detailed description from the tag wiki.")
                                    with gr.Box(visible=False) as results_tag_wiki_result_md_wrapper:
                                        results_tag_wiki_result_md = gr.Markdown()
                            
                            generate_btn = gr.Button("Generate New Image")

            with gr.Tab(label='Account'):
                with gr.Column(visible=True) as account_credentials_form:
                    with gr.Row():
                        account_forms_radio = gr.Radio([
                        "Login", "Register"], 
                        label='You are not currently logged in.', 
                        info='Please select account action:',
                        value="Login")
                        account_input_username_tb = gr.Textbox(label="Username", type="text")
                        account_input_password_tb = gr.Textbox(label="Password", type="password")
                    account_login_btn = gr.Button("Login", visible=True)
                    account_register_btn = gr.Button("Register", visible=False)
                
                with gr.Column(visible=False) as account_details_wrapper:
                    account_username = gr.Markdown()
                    account_experience = gr.Markdown()
                    account_questions = gr.Markdown()
                    account_accuracy = gr.Markdown()

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
                    results_tag_wiki_search_dropdown,
                    results_tag_wiki_result_md,
                    results_tag_wiki_result_md_wrapper,
                    generate_btn
                ]
            )

            results_tag_wiki_search_dropdown.change(
                fn=self.ui_update_tag_wiki_result,
                inputs=results_tag_wiki_search_dropdown,
                outputs=[
                    results_tag_wiki_result_md,
                    results_tag_wiki_result_md_wrapper
                ]
            )

            account_forms_radio.change(
                fn=self.ui_account_login_or_register_display,
                inputs=account_forms_radio,
                outputs=[
                    account_login_btn,
                    account_register_btn
                ]
            )

            # Use helper function to bind events
            def bind_account_auth_click_event(button: gr.Button, fn):
                button.click(
                    fn=fn,
                    inputs=[
                        account_input_username_tb,
                        account_input_password_tb,
                        gr_state
                    ],
                    outputs=[
                        gr_state,
                        account_credentials_form,
                        account_details_wrapper,
                        main_tab_user_name_md,
                        main_tab_user_exp_md,
                        main_tab_user_accuracy_md,
                        main_tab_user_questions_count_md
                    ]
                )
            bind_account_auth_click_event(account_register_btn, self.account_register)
            bind_account_auth_click_event(account_login_btn, self.account_login)

        demo.queue(concurrency_count=20).launch()
