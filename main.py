from app import App
from gui import GradioUI

def main():
    app = App(debug_mock_tags=False, debug_mock_image=False)
    GradioUI(app).launch()

if __name__ == '__main__':
    main()
