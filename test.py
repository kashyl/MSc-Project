import gradio as gr
from gui_timer import TimerApp

def launch():
    app = TimerApp()

    with gr.Blocks(css="footer {visibility: hidden}") as demo:
        start_btn = gr.Button('Start Timer')
        stop_btn = gr.Button('Stop Timer')
        text = gr.Textbox(value='')

        start_btn.click(fn=app.start_timer, outputs=text)
        stop_btn.click(fn=app.stop_timer, outputs=text)
    demo.queue(concurrency_count=20).launch()

launch()
