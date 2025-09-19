# jsGPT
Train a custom transformer model in the frontend, powered by TensorFlowJS and WebGPU!

## Outline
The following architectures have been written to run on the Node.js backend (using C++ bindings), and can be found in the `models/` directory:
- `layerBigram.js`: the bigram model implemented using the custom layer API.
- `modelBigram.js`: the bigram model implemented using the model API.
- `layerGPT.js`: the NanoGPT model implemented using the custom layer API.
- `modelGPT.js`: the NanoGPT model implemented using the model API.

I have written both a basic bigram model architecture and the full NanoGPT architecture. This is useful for pedagogical purposes, as it is often easier to try and grasp the workings of the bigram model before moving to NanoGPT. Additionally, I have written each architecture as both a TensorflowJS custom layer and a TensorflowJS model; if this project is ever useful to anybody, they can select which API is more suitable to their needs.

The `lab/` directory contains code for a simple website which allows users to experiment with training the models as implemented in TensorflowJS. The site runs on vanilla HTML, CSS, and JavaScript, and uses the WebGPU, WebGL, or WASM backends to allow for rapid execution of the training pipeline. The site is functioning and can be found [here](https://javascript-gpt.netlify.app/), though do note that webpage responsivity is currently being worked on.

## Purpose
1. To assist with my assigned research project, which involves creating a pipeline for training LLMs in the shell.
2. To get more familiar with popular ML frameworks by porting components across them.
3. To create a point of reference for users looking to get started with TensorflowJS (which is lacking in online tutorials and documentation), as the code here can easily be compared to the original NanoGPT.
4. To integrate this into a basic web project and practice a little web development.

## References
This project is a reimplementation of NanoGPT, outlined by Andrej Karpathy in this [video](https://youtu.be/kCc8FmEb1nY?si=1riNNp8rxrVCGeZs).
Oleksii Trekhleb's [homemade-gpt](https://github.com/trekhleb/homemade-gpt-js) was also a helpful reference for this project. My project is (deliberately) a much simpler implementation, written with fewer custom classes and using only JavaScript instead of TypeScript.
