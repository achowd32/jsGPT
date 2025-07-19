# jsGPT
A basic implementation of NanoGPT written in TensorflowJS.

## Outline
I intend to write both a basic bigram model architecture and the full NanoGPT architecture. This is useful for pedagogical purposes, as it is often easier for beginners to try and grasp how the bigram model works (both conceptually and as implemented in TensorflowJS) before moving to NanoGPT.

I also intend to write each architecture as both a TensorflowJS custom layer and a TensorflowJS model; if this project is ever useful to anybody, they can select which API is more suitable to their needs.

So far, the following architectures have been implemented:
- `layerBigram.js`: the bigram model implemented using the custom layer API.
- `modelBigram.js`: the bigram model implemented using the model API.
- `layerGPT.js`: the NanoGPT model implemented using the custom layer API.

## Purpose
1. To assist with my assigned research project, which involves creating a pipeline for training LLMs in the shell.
2. To get more familiar with popular ML frameworks by porting components across them.
3. To create a point of reference for users looking to get started with TensorflowJS (which is lacking in online tutorials and documentation), as the code here can easily be compared to the original NanoGPT.
4. To hopefully integrate this into a web project and practice a little web development.

## References
This project is a reimplementation of NanoGPT, outlined by Andrej Karpathy in this [video](https://youtu.be/kCc8FmEb1nY?si=1riNNp8rxrVCGeZs).
