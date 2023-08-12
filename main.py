from app import App
from gui import GradioUI

def main():
    app = App(debug_mock_image=True, debug_mock_tags=True)
    GradioUI(app).launch(share=True)

if __name__ == '__main__':
    main()
