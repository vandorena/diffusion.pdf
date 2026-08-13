<img src="./assets/logo.svg" width="300px">

> Run a diffusion model inside a PDF file.

This project is a fork of [EvanZhouDev's llm.pdf](https://github.com/EvanZhouDev/llm.pdf),
which runs a language model inside a PDF the same way. Watch how the original
was built [on YouTube](https://youtu.be/4cBom2lAx-g).

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

## Inspiration and Credits

Thank you to the following for inspiration and reference:
* [EvanZhouDev's llm.pdf](https://github.com/EvanZhouDev/llm.pdf), the project this repo is forked from
* [ading2210's DoomPDF](https://github.com/ading2210/doompdf)
* [rahuldshetty's llm.js](https://github.com/rahuldshetty/llm.js)