import re, random, os, hmac
import gradio as gr
from datetime import datetime
from dotenv import load_dotenv
from app import App
from shared import SDModels, RANDOM_MODEL_OPT_STRING, ObserverContext, DifficultyLevels, GUIFiltersLabels, DIFFICULTY_LEVEL_EXP_GAIN
from db_manager import GUIAlertType, DBResponse, UserState, QuestionKeys, get_default_state
from custom_logging import logger
from concurrent.futures import ThreadPoolExecutor
from prompt_generation.prompt_generation import PROMPT_CATEGORIES_FUNCTIONS, generate_prompt

load_dotenv()   # for app auth if share==True

DEFAULT_GUEST_WELCOME = ('**👤 Playing as a Guest.** Log in or **sign up** to gain access to **play statistics**, '
                                '**question history**, and **compete on leaderboards**!'
)
DROPDOWN_NO_SELECTION = "No Selection"
PROMPT_RANDOM_CATEGORY = "Random Category"


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
        # Process prompt
        if not prompt or prompt == PROMPT_RANDOM_CATEGORY:
            prompt = random.choice(list(PROMPT_CATEGORIES_FUNCTIONS.keys()))

        # If the prompt selection is one of the categories, generate.
        # Otherwise keep as is since it's a user custom prompt.
        if prompt in PROMPT_CATEGORIES_FUNCTIONS:
            prompt = generate_prompt(prompt)
            prompt = ', '.join(prompt)

        if not sd_model:
            sd_model = RANDOM_MODEL_OPT_STRING

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
        if display_btn is None:
            display_btn = False
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
        return gr.Row.update(visible=display_btn)

    def on_submit(self, state: dict, selected_tags: list, rating_filter: str):
        state_user = state[UserState.NAME]
        if state_user:
            self.app.submit_answer(selected_tags_list=selected_tags, username=state_user)
            state = self.app.db_manager.get_user_state(state, user_model_or_name=state_user)
        else:
            self.app.submit_answer(selected_tags_list=selected_tags)
        
        return (
            state,
            self.ui_update_submit_btn_display(False),
            self.ui_show_image_tags(False),
            self.ui_show_results_wrapper(),
            self.ui_update_results_markdown(),
            self.ui_update_tag_wiki_search(),
            self.ui_clear_tag_wiki_result(),
            self.ui_show_tag_wiki_result_wrapper(False),
            self.ui_update_generate_btn_display(True),
            *self.ui_update_user_account_information(state)
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
        
        if username in ['None', 'Guest', 'Admin']:  # usually there'd be a list of these but no need atm
            gr.Warning(f'Username {username} is restricted.')
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
        if response is None or response.state == get_default_state():
            return (
                get_default_state(),
                self.ui_display_column(True),
                self.ui_display_column(False),
                *self.ui_clear_user_account_information()
            )
        return (
            response.state, 
            self.ui_display_column(False),                              # account login/register form
            self.ui_display_column(True),                               # account details wrapper
            *self.ui_update_user_account_information(response.state)
        )

    def ui_update_user_account_information(self, state: dict):
        if state == get_default_state():
            return self.ui_clear_user_account_information()
        return (
            *self.ui_account_update_user_stats(state),                      # for the main tab
            *self.ui_account_update_user_stats(state),                      # for the account details
            self.ui_account_update_greeting(state),                         # for the account tab welcome message
            self.ui_account_update_question_history_selector(state)         # for the question history selector dropdown
        )

    def ui_clear_user_account_information(self):
        return (
            *self.ui_reset_main_tab_user_info(),                    # for the main tab
            *[gr.Markdown.update(value=None) for _ in range(4)],    # for the account details
            gr.Markdown.update(value=None),                         # for the account tab greeting message
            gr.Dropdown.update(choices=None)                        # for the question history selector dropdown
        )

    def ui_account_update_user_stats(self, state):
        # Extract the values from the state
        username = state[UserState.NAME]
        level, exp_for_current_level, exp_required_for_next_level = self.app.get_level_info(state[UserState.EXP])
        accuracy = state[UserState.ACCURACY]
        ans_count = state[UserState.ATTEMPTED_COUNT]

        # Create the markdown strings
        user_md = f'👤 Logged in as **{username}**'
        exp_md = f"**🏅 Level:** {level} | **✨ XP:** {exp_for_current_level}/{exp_required_for_next_level}"
        accuracy_md = f"**🎯 Accuracy rating:** {accuracy}%"
        ans_md = f"**❓ Questions attempted:** {ans_count}"

        return (
            gr.Markdown.update(value=user_md),
            gr.Markdown.update(value=exp_md, visible=True),
            gr.Markdown.update(value=accuracy_md, visible=True),
            gr.Markdown.update(value=ans_md, visible=True)
        )
        
    def ui_reset_main_tab_user_info(self):
        return (
            gr.Markdown.update(value=DEFAULT_GUEST_WELCOME),
            *[gr.Markdown.update(value=None, visible=False) for _ in range(3)]
        )

    def ui_account_update_greeting(self, state):
        return gr.Markdown.update(value=f"""
            # Hi there, {state[UserState.NAME]}! 👋

            This is your Account profile page. Here you can:

            - View your play statistics to track your performance.
            - Browse the gallery of images from the questions you've attempted.
            - Select and review any of your previously attempted questions.
            """)

    def process_image_and_generate_label(self, question_data, rating_filter):
        """Generate label and process the image based on the rating filter."""
        label = f"QID: {question_data[QuestionKeys.ID]}"
        image_path = question_data[QuestionKeys.IMAGE_FILE_PATH]
        image_rating = question_data[QuestionKeys.IMAGE_RATING]

        if self.app.content_filter.is_rating_filtered_gui(rating_filter, image_rating):
            label += f" ({question_data[QuestionKeys.IMAGE_RATING]})"
            image_path = self.app.content_filter.get_blurred_image_from_path(image_path)

        return image_path, label

    def _process_gallery_image(self, args):
        """ Helper function to process each image in parallel."""
        q, app, rating_filter = args  # unpack arguments
        return self.process_image_and_generate_label(q, rating_filter)

    def ui_account_update_image_gallery(self, state, rating_filter: str):        
        questions = state[UserState.ATTEMPTED_QUESTIONS]

        # Use ThreadPoolExecutor to process images concurrently
        with ThreadPoolExecutor() as executor:
            args_list = [(q, self.app, rating_filter) for q in questions]
            gallery_data = list(executor.map(self._process_gallery_image, args_list))
                    
        return gr.Gallery.update(value=gallery_data)
    
    def ui_account_update_question_history_selector(self, state):
        questions = state[UserState.ATTEMPTED_QUESTIONS]
        
        # Format date and time for each question
        dd_choices = [DROPDOWN_NO_SELECTION] + [
            f'Question ID: {q[QuestionKeys.ID]}, '
            f'Attempted on: {q[QuestionKeys.ATTEMPTED_TIME].strftime("%Y-%m-%d %H:%M:%S")}'
            for q in questions
        ]
        
        return gr.Dropdown.update(choices=dd_choices)

    @staticmethod
    def _extract_question_id(question_string: str) -> int:
        match = re.search(r"Question ID: (\d+)", question_string)
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"Invalid question string format: {question_string}")

    def ui_account_update_selected_previous_question_details(self, state, selected_question, rating_filter):
        if selected_question is None or selected_question == DROPDOWN_NO_SELECTION:
            return (
                gr.Box.update(visible=False),
                gr.Image.update(value=None, label=None),
                gr.Markdown.update(value=None),
                gr.Markdown.update(value=None),
                gr.Dropdown.update(choices=None)
            )

        question_data = self.app.db_manager.fetch_question(self._extract_question_id(selected_question))

        return (
            gr.Box.update(visible=True),     # for account_past_question_wrapper
            self.ui_update_past_question_image(question_data, rating_filter),
            self.ui_update_past_question_image_generation_info(question_data),
            self.ui_update_past_question_tags(state, question_data),
            self.ui_update_past_question_tag_wiki_search(question_data)
        )
        # display tag search dropdown with list of all image tags

    def ui_update_past_question_image(self, question_data, rating_filter):
        label = f"Question {question_data[QuestionKeys.ID]}"

        image, label = self.process_image_and_generate_label(question_data, rating_filter)
        return gr.Image.update(value=image, label=label)

    def ui_update_past_question_image_generation_info(self, question_data):
        return gr.Markdown.update(value=question_data[QuestionKeys.GENERATION_INFO])

    def ui_update_past_question_tags(self, state, question_data):
        # 1. Extract the correct and false tags from the question_data.
        correct_tags = question_data[QuestionKeys.CORRECT_TAGS]
        false_tags = question_data[QuestionKeys.FALSE_TAGS]
        question_id = question_data[QuestionKeys.ID]
        username = state[UserState.NAME]

        # 2. Fetch the selected tags for that question by the user.
        selected_tags = self.app.db_manager.fetch_question_user_answers(question_id=question_id, username=username)

        # Formatting the markdown
        md_string = "### Correct Tags\n"
        md_string += ", ".join([f"<b>{tag}</b> <span style='color: gray;'>({weight})</span>" for tag, weight in correct_tags])
        md_string += "\n\n### False Tags\n"
        md_string += ", ".join([f"<b>{tag}</b> <span style='color: gray;'>({weight})</span>" for tag, weight in false_tags])
        md_string += "\n\n### Selected Tags\n"
        
        if selected_tags:
            md_string += ", ".join([f"<span style='color: green;'>{tag}</span>" if tag in [ct[0] for ct in correct_tags] else 
                                    f"<span style='color: red;'>{tag}</span>" if tag in [ft[0] for ft in false_tags] else tag 
                                    for tag in selected_tags])
        else:
            md_string += "No tags selected."

        return gr.Markdown.update(value=md_string)

    def ui_update_past_question_tag_wiki_search(self, question_data):
        # Extract tags (both correct and false) from the question data
        correct_tags = self.app.tag_names_only(question_data[QuestionKeys.CORRECT_TAGS])
        false_tags = self.app.tag_names_only(question_data[QuestionKeys.FALSE_TAGS])

        # Populate the dropdown with combined tags
        all_tags = list(set(correct_tags + false_tags))
        return gr.Dropdown.update(choices=all_tags)

    def ui_update_past_question_tag_wiki_result(self, tag_name: str):
        if tag_name is None:
            return (
                gr.Markdown.update(value=None, visible=False),  # result markdown
                gr.Box.update(visible=False) 
            )
        # Retrieve tag info
        tag_title, tag_info = self.app.get_tag_wiki(tag_name)

        if not tag_info:
            no_info_msg = f"Couldn't retrieve information for the tag **{tag_name}**."
            return gr.Markdown.update(value=no_info_msg, visible=True)

        # Prepend the tag_title to the tag_info
        full_info = f"## {tag_title}\n\n{tag_info}"
        return (
            gr.Markdown.update(value=full_info, visible=True),  # result markdown
            gr.Box.update(visible=True)                         # box wrapper
        )

    def ui_handle_rating_filter_update(self, img_gallery: gr.Gallery, rating_filter, state, selected_question):
        """ Update past question gallery and image on content filter radio change. """

        if state == get_default_state():  # if user not logged in
            return (
                gr.Gallery.update(value=None),
                gr.Image.update(value=None, label=None)
            )
        
        if img_gallery:
            gallery_update = self.ui_account_update_image_gallery(state, rating_filter)
        else:
            gallery_update = gr.Gallery.update(value=None)

        # check if there's a valid selected question
        if selected_question and selected_question != DROPDOWN_NO_SELECTION:
            question_data = self.app.db_manager.fetch_question(self._extract_question_id(selected_question))
            image_update = self.ui_update_past_question_image(question_data=question_data, rating_filter=rating_filter)
        else:
            # if no selected question, then default image update
            image_update = gr.Image.update(value=None, label=None)

        return gallery_update, image_update

    def account_logout(self, state: dict):
        # All the components in the account details wrapper should be cleared
        # on "Log out" ClearButton.click(), so no need to take care of them here
        logger.info(f'User "{state[UserState.NAME]}" has logged out.')
        gr.Info('Logged out.')
        state = get_default_state()   # Reset state to default
        return (
            state,
            self.ui_display_column(True),           # Shows account login/register form
            self.ui_display_column(False),          # Hides account details wrapper
            *self.ui_reset_main_tab_user_info(),    # Clear all account details
        )

    def _collect_nested_components(self, component):
        """Recursively obtain all nested children components from a component."""
        children_list = []

        if hasattr(component, 'children') and component.children:
            # Ensure that component.children is iterable (i.e., a list or tuple)
            children = component.children if isinstance(component.children, (list, tuple)) else [component.children]
            
            for child in children:
                # Skip adding to the list if child is a button component
                if not isinstance(child, (gr.Button, gr.ClearButton)):
                    children_list.append(child)
                    children_list.extend(self._collect_nested_components(child))

        return children_list

    def fetch_leaderboard_data(self):
        leaderboard_data_raw = self.app.db_manager.fetch_all_users_leaderboard_data()
        
        # Adjust the data for proper representation in the leaderboard
        leaderboard_data_adjusted = []

        for user_data in leaderboard_data_raw:
            experience = user_data[1]  # Extract the experience value

            # Get the level info for the user's experience
            level, _, _ = self.app.get_level_info(experience)

            # Extract and convert accuracy to float
            accuracy_str = user_data[2]
            accuracy_float = float(accuracy_str.strip('%'))

            # Reconstruct the user's data with adjusted values
            adjusted_data = [
                user_data[0],       # User
                level,              # Level
                accuracy_float,     # Accuracy
                user_data[3]        # Questions
            ]

            leaderboard_data_adjusted.append(adjusted_data)

        last_updated_msg=f"<span style='color: grey;'>Last updated at {datetime.now().strftime('%H:%M:%S')}</span>"

        return leaderboard_data_adjusted, gr.Markdown.update(value=last_updated_msg)

    def submit_feedback(self, state: dict, feedback_text: str):
        # TODO: limit submissions properly to avoid spam, for now just hiding elements

        if not feedback_text:
            gr.Warning('Please enter your comments or suggestions in the provided text box before submitting.')
            return self.ui_refresh_feedback_submit()

        self.app.submit_feedback(feedback_text, state[UserState.NAME] if state[UserState.NAME] else 'Guest')
        gr.Info('Thank you for your feedback!')
        return (
            gr.Textbox.update(value='', visible=False), # user innput
            gr.Button.update(visible=False),            # submit btn
            gr.Box.update(visible=True)                 # thank you msg
        )

    def ui_refresh_feedback_submit(self):
        return (
            gr.Textbox.update(value='', visible=True), # user innput
            gr.Button.update(visible=True),            # submit btn
            gr.Box.update(visible=False)               # thank you msg
        )

    def launch(self, share=False):
        with gr.Blocks(css="footer {visibility: hidden}") as demo:
            gr_state = gr.State(value=get_default_state())

            with gr.Tab('Main'):
                with gr.Box():
                    with gr.Row():
                        main_tab_user_name_md = gr.Markdown(value=(DEFAULT_GUEST_WELCOME))
                        main_tab_user_exp_md = gr.Markdown(visible=False)
                        main_tab_user_accuracy_md = gr.Markdown(visible=False)
                        main_tab_user_questions_count_md = gr.Markdown(visible=False)
                with gr.Row():
                    generation_prompt = gr.Dropdown(
                        choices=[PROMPT_RANDOM_CATEGORY] + list(PROMPT_CATEGORIES_FUNCTIONS.keys()),
                        value=PROMPT_RANDOM_CATEGORY,
                        allow_custom_value=True,
                        interactive=True,
                        multiselect=False,
                        label='Image Generation Prompt',
                        info='Choose a category to generate an image prompt, or manually input your own description.'
                    )
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
                                with gr.Accordion('Generation Details & Submit Feedback', visible=False, open=False) as gen_info_wrapper:
                                    gen_info = gr.Markdown()
                                    main_user_feedback_txt = gr.Textbox(
                                        lines=3, 
                                        label="Suggestions or Issues", 
                                        info="Let us know if you have any suggestions or you would like to report an issue.",
                                        interactive=True
                                    )
                                    main_user_feedback_submit = gr.Button('Submit Feedback')
                                    with gr.Box(visible=False) as main_feedback_thank_you:
                                        gr.Markdown('Your feedback has been submitted. Thank you!')
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
                            with gr.Row(visible=False) as submit_btn_wrapper:
                                clear_selected_tags = gr.ClearButton(value="Clear Selection", components=image_tags)
                                submit_btn = gr.Button("Submit")

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
                        with gr.Row():
                            account_input_username_tb = gr.Textbox(label="Username", type="text", interactive=True)
                            account_input_password_tb = gr.Textbox(label="Password", type="password", interactive=True)
                    account_login_btn = gr.Button("Login", visible=True)
                    account_register_btn = gr.Button("Register", visible=False)
                
                with gr.Column(visible=False) as account_details_wrapper:
                    with gr.Row():
                        with gr.Column():
                            with gr.Box():
                                account_greeting = gr.Markdown()
                            with gr.Box():
                                account_username = gr.Markdown()
                                account_experience = gr.Markdown()
                                account_questions = gr.Markdown()
                                account_accuracy = gr.Markdown()
                            with gr.Row():
                                account_update_gallery_btn = gr.Button('Update Image Gallery')                        
                        account_past_questions_image_gallery = gr.Gallery(label='Generated Images History', columns=4, rows=1)

                    account_past_questions_selector_dd = gr.Dropdown(
                        label='Question History', 
                        info="Select a previous question to review.",
                        interactive=True,
                        allow_custom_value=False,
                        scale=3
                        )

                    with gr.Box(visible=False) as account_past_question_wrapper:
                        with gr.Row():
                            with gr.Column():
                                account_past_question_image = gr.Image(show_download_button='True', show_share_button='True')
                                with gr.Accordion('Generation Details', open=False):
                                    account_past_question_image_gen_info = gr.Markdown()

                                # # TODO: a gr.Button() elem defined before the gr.ClearButton() seems to cause issues.
                                # #       Low priority issue, fix later if there is time.
                                #     account_user_feedback_txt = gr.Textbox(
                                #         lines=3, 
                                #         label="Suggestions or Issues", 
                                #         info="Let us know if you have any suggestions or you would like to report an issue.",
                                #         interactive=True
                                #     )
                                #     account_user_feedback_submit = gr.Button('Submit Feedback')
                                #     with gr.Box(visible=False) as account_feedback_thank_you:
                                #         gr.Markdown('Your feedback has been submitted. Thank you!')

                            with gr.Column():
                                with gr.Box():
                                    account_past_question_tags_md = gr.Markdown()
                                account_past_question_tag_search = gr.Dropdown(
                                        allow_custom_value=False, 
                                        multiselect=False,
                                        label="Tag Details Lookup",
                                        info="Select a tag to retrieve its detailed description from the tag wiki.")
                                with gr.Box(visible=False) as past_question_tag_result_wrapper:
                                    account_past_question_tag_search_result = gr.Markdown()
                    with gr.Row():
                        account_logout_btn = gr.ClearButton(
                            value="Log out",
                            components = [
                                *self._collect_nested_components(account_details_wrapper),
                                account_input_username_tb,
                                account_input_password_tb,
                            ]
                        )
                        
            with gr.Tab(label='Leaderboard'):
                with gr.Row():
                    with gr.Column(scale=7):
                        gr.Markdown(
                            '''
                            ## Leaderboard 
                            The top 100 players by level will be showcased here. 
                            The leaderboard updates automatically as you progress. 
                            Alternatively, you can manually update it by pressing the Refresh button.
                            '''
                        )
                    with gr.Row():
                        with gr.Box():
                            leaderboard_last_updated_msg = gr.Markdown(
                                value=f"<span style='color: grey;'>Last updated at {datetime.now().strftime('%H:%M:%S')}</span>"
                            )
                        leaderboard_refresh_btn = gr.Button('Refresh')
                leaderboard = gr.Dataframe(
                    headers=['👤 User', '🏅 Level', '🎯 Accuracy', '❓ Questions Answered'],
                    datatype=["str", "number", "number", "number"],
                    col_count=(4, 'fixed'),
                    max_rows=100,   # show top 100 players
                    interactive=False,
                    value=lambda: self.fetch_leaderboard_data()[0],
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
            def bind_generate_click_event(button: gr.Button):
                button.click(
                    self.generate_question, 
                    inputs=[
                        generation_prompt, 
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
                        submit_btn_wrapper,
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
                    submit_btn_wrapper
                ]
            )

            main_tab_user_details_components = [
                main_tab_user_name_md,
                main_tab_user_exp_md,
                main_tab_user_accuracy_md,
                main_tab_user_questions_count_md
            ]
            account_tab_user_details_components = [
                account_username,
                account_experience,
                account_accuracy,
                account_questions
            ]
            user_data_components = [
                *main_tab_user_details_components,
                *account_tab_user_details_components,
                account_greeting,
                account_past_questions_selector_dd
            ]

            submit_btn.click(
                fn=self.on_submit, 
                inputs=[
                    gr_state,
                    image_tags,
                    content_filter_rating
                ],
                outputs=[
                    gr_state,
                    submit_btn_wrapper,
                    image_tags,
                    results_wrapper,
                    results_md,
                    results_tag_wiki_search_dropdown,
                    results_tag_wiki_result_md,
                    results_tag_wiki_result_md_wrapper,
                    generate_btn,
                    *user_data_components
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
                        *user_data_components
                    ]
                )
            bind_account_auth_click_event(account_register_btn, self.account_register)
            bind_account_auth_click_event(account_login_btn, self.account_login)

            account_logout_btn.click(
                fn=self.account_logout,
                inputs=[gr_state],
                outputs=[                             # Account Details will be auto cleared by ClearButton.click()
                    gr_state,
                    account_credentials_form,         # Shows back the authentication form
                    account_details_wrapper,          # Hides account details
                    main_tab_user_name_md,            # Clears the main tab username info
                    main_tab_user_exp_md,             # Clears the main tab experience info
                    main_tab_user_accuracy_md,        # Clears the main tab accuracy info
                    main_tab_user_questions_count_md  # Clears the main tab question count info
                ]
            )

            account_update_gallery_btn.click(
                fn=self.ui_account_update_image_gallery,
                inputs=[
                    gr_state,
                    content_filter_rating
                ],
                outputs=[
                    account_past_questions_image_gallery
                ]
            )

            account_past_questions_selector_dd.change(
                fn=self.ui_account_update_selected_previous_question_details,
                inputs=[
                    gr_state,
                    account_past_questions_selector_dd,
                    content_filter_rating
                ],
                outputs=[
                    account_past_question_wrapper,
                    account_past_question_image,
                    account_past_question_image_gen_info,
                    account_past_question_tags_md,
                    account_past_question_tag_search
                ]
            )

            account_past_question_tag_search.change(
                fn=self.ui_update_past_question_tag_wiki_result,
                inputs=account_past_question_tag_search,
                outputs=[
                    account_past_question_tag_search_result,
                    past_question_tag_result_wrapper
                ]
            )

            content_filter_rating.change(
                fn=self.ui_handle_rating_filter_update,
                inputs=[
                    account_past_questions_image_gallery,
                    content_filter_rating,
                    gr_state,
                    account_past_questions_selector_dd
                ],
                outputs=[
                    account_past_questions_image_gallery,
                    account_past_question_image
                ]
            )

            leaderboard_refresh_btn.click(
                fn=self.fetch_leaderboard_data,
                outputs=[leaderboard, leaderboard_last_updated_msg]
            )

            main_tab_user_exp_md.change(
                fn=self.fetch_leaderboard_data,
                outputs=[leaderboard, leaderboard_last_updated_msg]
            )

            # User feedback
            def assign_user_feedback_events(button, text_input, thank_you, gen_info_obj=None):
                # click event
                button.click(
                    fn=self.submit_feedback,
                    inputs=[
                        gr_state,
                        text_input
                    ],
                    outputs=[
                        text_input,
                        button,
                        thank_you
                    ]
                )
                # change event if gen_info_obj provided
                if gen_info_obj:
                    gen_info_obj.change(
                        fn=self.ui_refresh_feedback_submit,
                        outputs=[
                            text_input,
                            button,
                            thank_you
                        ]
                    )
            configs = [
                (main_user_feedback_submit, main_user_feedback_txt, main_feedback_thank_you, gen_info),
                # (account_user_feedback_submit, account_user_feedback_txt, account_feedback_thank_you, account_past_question_image_gen_info)
            ]
            for conf in configs:
                assign_user_feedback_events(*conf)

        def app_auth(username, password):
            auth_user = os.getenv('APP_AUTH_USER')
            auth_pwd = os.getenv('APP_AUTH_PWD')

            if auth_user is None or auth_pwd is None:
                raise EnvironmentError("Required environment variables are not set.")
            
            # constant-time comparison to prevent timing attacks
            return hmac.compare_digest(username, auth_user) and hmac.compare_digest(password, auth_pwd)

        demo.queue(concurrency_count=20).launch(
            share=share, 
            server_port=7868, 
            auth=app_auth if share else None, 
            auth_message='Please enter the login details sent alongside the share link.'
        )
