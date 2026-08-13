<img src="./assets/logo.svg" width="300px">

# A PDF that generates ✨Sh*tty Images✨

## It has a f****** diffusion image model embedded into it!!!

[Download diffusion.pdf (75MB)](https://github.com/vandorena/diffusion.pdf/releases/download/v1/diffusion-chars.pdf) &nbsp;|&nbsp; [GitHub Repo](https://github.com/vandorena/diffusion.pdf)

## So how did I get here? :D

The last two weeks i've been messing around with pdfs. I made [slop pdf](https://slop.alexvd.dev), a pdf that generates random ai slop stories when you open it, and I was showing that off to one of my friends who suggested I get the LLM to actually work inside the pdf, without an API call to a relay.

I laughed at the idea, but then we did some research together, and we managed to find EvanZhouDev's [llm.pdf](https://github.com/EvanZhouDev/llm.pdf). Because it existed, I didn't want to waste my time or my tokens building something like it. So I pivoted, eventually deciding to make a diffusion image model work fully inside a pdf.

I forked EvanZhouDev's llm.pdf repo, and eventually managed to get it working (with the help of claude). Here's the finished version.

> **proof of conceptness disclaimer/warning :p**
>
> This model only has **131 Known Objects**. If you don't use one of these objects, it will try to map your input to a known object, terribly.

## Requirements

* **Chrome** or any other chromium browser (minus safari) -- plz note no safari/firefox support
* at least 75MB of storage (sorry its such a big file 😭)

## So is this AI?

Ehh.. it is technically a **denoising diffusion model** similar to Stable Diffusion, Midjourney or Dall-e but like way worse

I trained this model on 130ish categories using Google's Quick Draw dataset of doodle images. It took me so so long (because I did it on my Macbook Air) but eventually I trained it on enough data, for it to produce recognisable stuff.

This training took alot, and ended up inflating the PDF's size to 78mb :cry:.

Now you may be wondering, how'd i get the noise?.

I get the noise by using the current time, legit the only way to get something noisy from a pdf. ;P

## The Specs

| | |
|---|---|
| # of Parameters | 55,403,136 int8 |
| PDF Size | 75 MB |
| Vocabulary Size | 131 doodle categories |
| Compute | 886M MAC per image |
| Speed | ~15.3s per image (at 58M MAC/s) |
| Quality | 0.874 class accuracy |
| Training | 45 epochs, 327,500 drawings, final loss 0.2476 |

## Roll the Credits

Built by me, **[Alex Van Doren](https://alexvd.dev/)**. I'm one of the developers behind [Hack Club's Stardance Challenge](https://stardance.hackclub.com/), I'm a super cool sailor, and I'm also an incoming freshman at Brown University.

This is a fork of the really cool [llm.pdf](https://github.com/EvanZhouDev/llm.pdf) by [Evan Zhou](https://github.com/EvanZhouDev), that runs an LLM inside a PDF, the thing I was originally intending to build. Make sure to check out his [Youtube Video](https://youtu.be/4cBom2lAx-g) about it, which honestly explained alot.

I also wanted to credit [linuxpdf](https://github.com/ading2210/linuxpdf) and [DoomPDF](https://github.com/ading2210/doompdf) by [ading2210](https://github.com/ading2210/). They inspired me to experiment within the PDF spec, and the coolest thing about them is that they were actually submitted to a Hack Club program I worked on, [The Summer of Making](https://summer.hackclub.com/).

Also ty Google, for making the [Quick, Draw!](https://quickdraw.withgoogle.com/data) dataset.

Copyright © 2026 vandorena. Source on [GitHub](https://github.com/vandorena/diffusion.pdf). | Check out my personal site [alexvd.dev](https://alexvd.dev/)
