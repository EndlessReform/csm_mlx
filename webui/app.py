import gradio as gr
import numpy as np
from csm_mlx.loaders import CSM, Segment
from csm_mlx.lm.utils.text import TextNormalizer
from typing import Tuple, Optional
from scipy.signal import resample
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Argument parser for code quantization"
    )

    parser.add_argument(
        "--num_codes", type=int, default=32, help="Number of codes (default: 32)"
    )

    parser.add_argument(
        "--quantize",
        type=str,
        choices=["f32", "bf16", "q8"],
        default="bf16",
        help="Quantization type (default: bf16, choices: f32, bf16, q8)",
    )
    parser.add_argument("--model-id", type=str, default="jkeisling/csm-1b")

    return parser.parse_args()


args = get_args()

model = CSM(model_id=args.model_id, depth=args.num_codes, quantization=args.quantize)
model.warmup()

normalizer = TextNormalizer()


def synthesize_speech(
    text,
    temperature,
    min_p,
    voice: Optional[Tuple[int, np.ndarray]],
    transcript: str,
):
    if transcript is None and voice is not None:
        gr.Warning("Must have transcript for reference audio!")
        return
    """Generate speech from text using Fish Speech, processing each sentence separately."""
    pcm_list = []

    if voice is None:
        gr.Warning("Generating without reference audio!")
        context = []
    else:
        sr, sample = voice
        # Crude cache clear
        sample = sample.astype(np.float32) / 32768.0
        if sr != 24_000:
            num_samples = int(len(sample) * 24_000 / sr)
            sample = resample(sample, num_samples)

        context = [Segment(speaker=0, text=transcript, audio=sample)]

    # Generate audio for each sentence individually

    # Split the text into sentences
    sentences = normalizer.sentenceize(text)
    pcm_list = []

    # Generate audio for each sentence individually
    for i, sentence in enumerate(sentences):
        pcm = model(
            text=sentence,
            speaker_id=0,
            context=context,
            temp=temperature,
            # TODO kv caching
            # Screw it, do a proper ring buffer later
            use_last_gens=i > 0,
            keep_prompt_only=True,
            backbone_min_p=min_p,
        )
        pcm_list.append(pcm.flatten())
        # Add 250ms of silence in between sentences
        silence = np.zeros(int(0.25 * 24_000))
        pcm_list.append(silence)

    # Concatenate all PCM arrays into one
    final_pcm = np.concatenate(pcm_list)
    return (24_000, final_pcm)


# Create the Gradio interface
with gr.Blocks(
    theme=gr.themes.Default(
        font=[gr.themes.GoogleFont("IBM Plex Sans"), "Arial", "sans-serif"],
        font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.slate,
    )
) as demo:
    with gr.Row():
        gr.Markdown("""
            # CSM 1B
            """)

    with gr.Row():
        with gr.Column():
            voice_input = gr.Audio(label="Voice Input", type="numpy", sources="upload")
            transcript_text = gr.Textbox(
                label="Transcript Text",
                placeholder="The text of your reference audio. Keep it short - 1-3 sentences only!",
                lines=3,
            )
            input_text = gr.Textbox(
                label="Input Text", placeholder="Enter text to synthesize...", lines=3
            )
            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.9, step=0.1, label="Temperature"
                )
                min_p = gr.Slider(
                    minimum=0.0, maximum=0.2, value=0.1, step=0.01, label="min-p"
                )
        with gr.Column():
            audio_output = gr.Audio(label="Generated Speech", type="numpy")

    generate_btn = gr.Button("Generate Speech", variant="primary")
    generate_btn.click(
        fn=synthesize_speech,
        inputs=[input_text, temperature, min_p, voice_input, transcript_text],
        outputs=[audio_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
