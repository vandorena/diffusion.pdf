<img src="./assets/logo.svg" width="300px">

> Run LLMs inside a PDF file. And now diffusion models too.

Watch how llm.pdf was built [on YouTube](https://youtu.be/4cBom2lAx-g).

## diffusion.pdf

Type a word, press Generate, and a **denoising diffusion model** runs sixteen
sampling steps in the PDF viewer's own JavaScript engine, painting the result
onto a 28x28 grid of form widgets. No network, no plugin, no external file.

6.3M parameters, quantised to int8 and base64'd into the document, trained on
[Quick, Draw!](https://quickdraw.withgoogle.com/data) doodles of sixteen
everyday objects. It is a real diffusion model — v-prediction, a cosine noise
schedule, deterministic DDIM sampling and classifier-free guidance — not a
single-pass decoder.

```sh
python3 -m pip install -r requirements.txt -r requirements-train.txt

python3 -m train.train_diffusion              # ~7 min, writes train/model.json
python3 scripts/generateDiffusionPDF.py --bake-initial
node tools/harness.mjs                        # proves the in-PDF JS matches numpy
```

The JavaScript that ships inside the PDF is verified against a numpy reference
implementation: the random stream matches exactly, the initial noise is
bit-identical, and the entire 16-step sampling trajectory agrees to 1.7e-14.
Full detail, including what is still unmeasured, in
[docs/diffusion.md](docs/diffusion.md).

**The honest ceiling:** this maps a typed word to the nearest of sixteen known
objects and draws it. It is not open-vocabulary generation, and it tells you in
the status line when it has guessed.

## What is llm.pdf?

This is a proof-of-concept project, showing that it's possible to run an entire Large Language Model in nothing but a PDF file.

It uses [Emscripten](https://emscripten.org/) to compile [llama.cpp](https://github.com/ggml-org/llama.cpp?tab=readme-ov-file) into [asm.js](https://en.wikipedia.org/wiki/Asm.js), which can then be run in the PDF using an old PDF JS injection.

Combined with embedding the entire LLM file into the PDF with base64, we are able to run LLM inference in nothing but a PDF.

[Watch the video on YouTube](https://youtu.be/4cBom2lAx-g) to learn the full story!

## Load a Custom Model in the PDF

The `scripts/generatePDF.py` file will help you create a PDF with any compatible LLM.

The easiest way to get started is with the following command:
```sh
cd scripts
python3 generatePDF.py --model "path/for/model.gguf" --output "path/to/output.pdf"
```

### Choosing a Model

Here's the general guidelines when picking a model:

* Only GGUF quantized models work.
* Generally, try to use Q8 quantized models, as those run the fastest.
* For reference, 135M parameter models take around 5s per token input/output. Anything higher will likely be unreasonably slow.

## Inspiration and Credits

Thank you to the following for inspiration and reference:
* [ading2210's DoomPDF](https://github.com/ading2210/doompdf)
* [rahuldshetty's llm.js](https://github.com/rahuldshetty/llm.js)

Thank you to the following for creating the tiny LLMs that power llm.pdf:
* [EleutherAI's pythia models](https://github.com/EleutherAI/pythia)
* [Ronen Eldan and Yuanzhi Li's TinyStories LLM](https://arxiv.org/abs/2305.07759)
* [arnir0's Tiny-LLM](https://arxiv.org/abs/2305.07759)