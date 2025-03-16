from fastrtc import ReplyOnPause, Stream
import gradio as gr
import numpy as np


def detection(image, slider):
    return np.flip(image, axis=0)


def response(audio: tuple[int, np.ndarray]):
    sr, pcm = audio


stream = Stream(
    handler=detection,  #
    modality="audio",  #
    mode="send-receive",  #
    additional_inputs=[
        gr.Slider(minimum=0, maximum=1, step=0.01, value=0.3)  #
    ],
    additional_outputs=None,  #
    additional_outputs_handler=None,  #
)

if __name__ == "__main__":
    stream.ui.launch()
