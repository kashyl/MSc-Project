import smtplib, datetime, random, string, os
from email.message import EmailMessage
from dotenv import load_dotenv
from custom_logging import logger

load_dotenv()
GMAIL_ADDRESS = os.getenv('GMAIL_ADDRESS')
GMAIL_APP_PASS = os.getenv('GMAIL_APP_PASS')

class UserFeedback:
    def __init__(self):
        pass

    def generate_ticket_number(self):
        # 1. Fixed Prefix
        prefix = "TKT"
        
        # 2. Current date and time
        date_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        
        # 3. Random sequence of characters
        random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
        
        ticket_number = f"{prefix}-{date_str}-{random_str}"
        return ticket_number

    def format_feedback_email(self, user_input_text, image_gen_info, user_name,
                              image_gen_time, image_rating, image_tags, false_tags,
                              difficulty_level, user_selected_tags):
        
        message_body = (
            f"User Feedback - Ticket {self.generate_ticket_number()}\n\n"
            f"User Input Text: {user_input_text}\n"
            f"Image Generation Info: {image_gen_info}\n"
            f"Username: {user_name if user_name else 'Anonymous'}\n"
            f"Image Generation Time: {image_gen_time}\n"
            f"Image Rating: {image_rating}\n"
            f"Image Tags: {image_tags}\n"
            f"False Tags: {false_tags}\n"
            f"Difficulty Level: {difficulty_level}\n"
            f"User Selected Tags: {user_selected_tags if user_selected_tags else 'Not Provided'}"
        )

        logger.info(f'User feedback submitted: {message_body}')

        return message_body

    def send_email(self, email_content):
        msg = EmailMessage()
        msg.set_content(email_content)

        # Setting up basic email parameters
        msg['Subject'] = f"Feedback Received - {self.generate_ticket_number()}"
        msg['From'] = GMAIL_ADDRESS
        msg['To'] = GMAIL_ADDRESS  # Sending to yourself

        try:
            # Using Gmail's SMTP server with SSL
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(GMAIL_ADDRESS, GMAIL_APP_PASS)
                smtp.send_message(msg)
            logger.info("User feedback email successfully sent.")

        except Exception as e:
            logger.error(f"Failed to send user feedback email. Error: {e}")

    def process_feedback(self, user_input_text, image_gen_info, user_name=None,
                         image_gen_time=None, image_rating=None, image_tags=None,
                         false_tags=None, difficulty_level=None, user_selected_tags=None):
        
        email_content = self.format_feedback_email(user_input_text, image_gen_info, user_name, 
                                                   image_gen_time, image_rating, image_tags, 
                                                   false_tags, difficulty_level, user_selected_tags)
        self.send_email(email_content)
