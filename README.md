# Sesame CSM 1B MLX port

Port of Sesame's [CSM](https://github.com/SesameAILabs/csm) model to [MLX](https://github.com/ml-explore/mlx) for use on Apple Silicon.

The project goal is realtime streaming inference on a MacBook.

## Installation

Clone this repo.

Get [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already. Then:

```bash
uv sync
```

Run the WebUI with:

```bash
uv run webui/app.py
```

Q8 quantization is available for a ~60% speedup at some loss of quality:

```bash
uv run webui/app.py --quantize q8
```

For procedural examples, check out `example.ipynb`

## Roadmap

- [x] Safetensors conversion
- [x] Core modeling and entry point
- [x] Gradio UI (blocking, streaming will be supported below)
- [x] Streaming output (for model)
- [ ] FastRTC speech-to-speech webui
- [ ] Perf improvements
- [ ] PyPI library
