from app import App
from gui import GradioUI

def main():
    app = App(debug_mock_image=False, debug_mock_tags=False)
    GradioUI(app).launch(share=True)

if __name__ == '__main__':
    main()
