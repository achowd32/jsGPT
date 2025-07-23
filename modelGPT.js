#!/usr/bin/env node
// WORK IN PROGRESS
import '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';

// hyperparameters
const BATCH_SIZE = 12;
const BLOCK_SIZE = 64;
const MAX_ITERS = 200;
const N_EMBD = 128;
const N_LAYER = 4;
const N_HEAD = 4;
const HEAD_SIZE = 16;
const LEARNING_RATE = 0.001;
const EVAL_ITERS = 50;
const DROPOUT = 0.0;

// read in data file
const dataStr = fs.readFileSync('data.txt').toString();

// set up token encoder and decoder
const charList = Array.from(new Set(dataStr)).sort();
const vocabSizeVal = charList.length;
const stoi = new Map(charList.map((val, ind) => [val, ind]));
const itos = new Map(charList.map((val, ind) => [ind, val]));
function encode(str){
  return Array.from(str).map(c => stoi.get(c));
}
function decode(toks){
  return toks.map(i => itos.get(i)).join('');
}

// set up training and validation data
const dataTensor = tf.tensor(encode(dataStr), undefined, "int32");
const trainSize = Math.round(0.9 * dataTensor.size);
const valSize = dataTensor.size - trainSize;

const trainTensor = dataTensor.slice([0], [trainSize]);
const valTensor = dataTensor.slice([trainSize], [valSize]);

// set up data loader
function getBatch(split){
  // establish which data to use
  let data = trainTensor;
  if(split === "val"){
    data = valTensor;
  }
  
  // indices to sample from
  let minInd = 0;
  let maxInd = data.size - BLOCK_SIZE; 
  const randInds = tf.randomUniform([BATCH_SIZE], minInd, maxInd, "int32").arraySync();

  // get samples
  const xRows = [];
  const yRows = [];
  for(let i = 0; i < BATCH_SIZE; i++){
    let curSplit = randInds[i];
    let xTensor = data.slice([curSplit], [BLOCK_SIZE]);
    let yTensor = data.slice([curSplit + 1], [BLOCK_SIZE]);
    xRows.push(xTensor);
    yRows.push(yTensor);
  }
  
  // use stack to convert to 2D tensor
  const xVal = tf.stack(xRows);
  const yVal = tf.stack(yRows);
  return {x: xVal, y: yVal};
}
// ------------------- LAYER DEFINITIONS ------------------------

// layer to perform the scaling operation in Head
class ScaleLayer extends tf.layers.Layer {
  constructor(config) {
    super(config);
    this.scale = config.scale;
  }
  call(inputs) {
    const x = Array.isArray(inputs) ? inputs[0] : inputs;
    return tf.mul(x, tf.scalar(this.scale));
  }
  getConfig() {
    return Object.assign(super.getConfig(), {scale: this.scale});
  }
  static get className() { return 'ScaleLayer'; }
}
tf.serialization.registerClass(ScaleLayer);

// layer to perform the masking operation in Head
class CausalMask extends tf.layers.Layer {
  constructor(config) {
    super(config);
    this.blockSize = config.blockSize;
  }
  build(inputShape) {
    this.tril = tf.linalg.bandPart(tf.ones([this.blockSize, this.blockSize], 'bool'), -1, 0);  // shape [Tmax, Tmax]
    super.build(inputShape);
  }
  call(inputs/*, kwargs*/) {
    const scores = Array.isArray(inputs) ? inputs[0] : inputs; // [B, T, T]
    const shape = scores.shape; // e.g. [1, 10, 10]
    const T = shape[1]; // 10 at runtime

    // slice out the [T, T] submatrix
    const mask2d = this.tril.slice([0, 0], [T, T]); // [T, T]
    const mask = mask2d.logicalNot().expandDims(0); // [1, T, T]

    const negInf = tf.fill(shape, Number.NEGATIVE_INFINITY); // [1, T, T]
    return tf.where(mask, negInf, scores); // [1, T, T]
  }
  getConfig() {
    return Object.assign(super.getConfig(), {blockSize: this.blockSize});
  }
  static get className() { return 'CausalMask'; }
}
tf.serialization.registerClass(CausalMask);

// ------------------- MODEL DEFINITIONS ------------------------

// define function that returns Identity layer as a tf.model
// can be used to replace the time intensive layerNorm operation
function createIdentity(nEmbd) {
  const input = tf.input({shape: [null, nEmbd]});
  // identity function - just return the input as output
  return tf.model({inputs: input, outputs: input});
}

// define function that returns a LayerNorm layer as a tf.model
function createLayerNorm(nEmbd) {
  const input = tf.input({shape: [null, nEmbd]});
  const normalized = tf.layers.layerNormalization().apply(input);
  return tf.model({inputs: input, outputs: normalized});
}

// define function that returns a Head layer as a tf.model
function createHead(blockSize, nEmbd, headSize, dropoutRate) {
  // create symbolic input
  const input = tf.input({shape: [null, nEmbd], dtype: 'float32'});

  // create key, query, value embeddings; (B, T, headSize)
  const key   = tf.layers.dense({units: headSize, useBias: false}).apply(input);
  const query = tf.layers.dense({units: headSize, useBias: false}).apply(input);
  const value = tf.layers.dense({units: headSize, useBias: false}).apply(input);

  // compute self attention scores
  const keyT = tf.layers.permute({dims: [2, 1]}).apply(key); // (B, headSize, T)
  const scores = tf.layers.dot({axes: [2, 1]}).apply([query, keyT]); // (B, T, headSize) @ (B, headSize, T) = (B, T, T)
  const scaled = new ScaleLayer({scale: 1/Math.sqrt(headSize)}).apply(scores); // scale by 1/sqrt(headSize)
  
  // apply mask
  const masked = new CausalMask({blockSize}).apply(scaled);
  const weights = tf.layers.activation({activation: 'softmax'}).apply(masked);

  // apply dropout
  const dropped = tf.layers.dropout({rate: dropoutRate}).apply(weights);

  // perform weighted aggregation of the values
  const out = tf.layers.dot({axes: [2, 1]}).apply([dropped, value]);

  // build the model
  return tf.model({inputs: input, outputs: out});
}

// define function that returns a FeedForward layer as a tf.model
function createFeedForward(nEmbd) {
  const input = tf.input({shape: [null, nEmbd]});
  
  // expansion layer with ReLU activation
  const expanded = tf.layers.dense({
    units: 4 * nEmbd,
    activation: 'relu',
  }).apply(input);
  
  // compression layer
  const compressed = tf.layers.dense({
    units: nEmbd,
  }).apply(expanded);
  
  // dropout layer
  const output = tf.layers.dropout({rate: DROPOUT}).apply(compressed);
  return tf.model({inputs: input, outputs: output});
}

// function that returns MultiHeadAttention as a tf.model
function createMultiHeadAttention(numHeads, headSize, nEmbd, blockSize, dropRate) {
  const input = tf.input({shape: [null, nEmbd]});

  // instantiate heads
  const heads = Array.from({length: numHeads},
    () => createHead(blockSize, nEmbd, headSize, dropRate));

  // apply each head in parallel
  let out = heads.map(head => head.apply(input));

  // concat each headOut value along the feature axis 
  out = tf.layers.concatenate({axis: 2}).apply(out);

  // apply projection layer
  out = tf.layers.dense({units: nEmbd}).apply(out);

  // apply dropout
  out = tf.layers.dropout({rate: dropRate}).apply(out);

  return tf.model({inputs: input, outputs: out});
}

// define Transformer block
class Block extends tf.layers.Layer{
  constructor(nEmbd, nHead){
    super({});
    this.nEmbd = nEmbd;
    this.nHead = nHead;
    this.headSize = Math.floor(nEmbd / nHead);
  }

  build(){
    // create self attention layer
    this.sa = createMultiHeadAttention(this.nHead, this.headSize, this.nEmbd, BLOCK_SIZE, DROPOUT);

    // create feed forward layer
    this.ffwd = createFeedForward(this.nEmbd);

    // create layerNorm layers, or use the Identity layer to avoid layerNorm
    // this.ln1 = createLayerNorm(this.nEmbd);
    // this.ln2 = createLayerNorm(this.nEmbd);
    this.ln1 = createIdentity(this.nEmbd);
    this.ln2 = createIdentity(this.nEmbd);

    super.build();
  }

  call(input){
    // perform computations with residual
    let out = input.add(this.sa.apply(this.ln1.apply(input))); // input + sa(ln1(input))
    out = out.add(this.ffwd.apply(this.ln2.apply(out))); // out + ffwd(ln2(out))
    return out;
  }
  
  getClassName(){ return 'Block'; }
}

// define GPT language model
class GPTLanguageModel extends tf.layers.Layer {
  constructor(){
    super({});
    this.vocabSize = vocabSizeVal;
    this.nLayer = N_LAYER;
    this.nEmbd = N_EMBD;
    this.nHead = N_HEAD;
    this.blockSize = BLOCK_SIZE;
  }

  build(){
    // build token embedding table
    this.tokenEmbeddingTable = tf.layers.embedding({
      inputDim: this.vocabSize,
      outputDim: this.nEmbd,
    });
    this.tokenEmbeddingTable.build([null, this.blockSize]);

    // build position embedding table
    this.positionEmbeddingTable = tf.layers.embedding({
      inputDim: this.blockSize,
      outputDim: this.nEmbd,
    });
    this.positionEmbeddingTable.build([null, this.blockSize]);

    // array of transformer blocks
    this.blockArr = [];
    for(let i  = 0; i < this.nLayer; i++){
      const blk = new Block(this.nEmbd, this.nHead);
      blk.build();
      this.blockArr.push(blk);
    }

    // build final layernorm
    this.ln = tf.layers.layerNormalization();

    // build linear layer
    this.lmHead = tf.layers.dense({
      inputDim: this.nEmbd,
      units: this.vocabSize,
    });

    super.build();
  }

  call(inputs){ // FIX (CHECK) DIMENSIONS
    // get input shape
    const [B, T] = inputs.shape;

    // get embeddings as a sum of token and position embeddings
    const tokEmbd = this.tokenEmbeddingTable.apply(inputs); // (B, T, nEmbd)
    const posEmbd = this.positionEmbeddingTable.apply(
      tf.range(0, T, 1, "int32")).expandDims(0); // (1, T, nEmbd)
    const embdSum = tokEmbd.add(posEmbd); // (B, T, nEmbd)

    // apply all transformer blocks sequentially
    let blockEmbd = embdSum; // (B, T, nEmbd)
    for(const block of this.blockArr){
      blockEmbd = block.apply(blockEmbd);
    }
    blockEmbd = this.ln.apply(blockEmbd);

    const logits = this.lmHead.apply(blockEmbd); // (B, T, vocabSize)
    return logits;
  }
  
  loss(inputs, targets){
      // get logits
      const logitsT = this.apply(inputs);

      // flatten logits and targets
      const flatLogits = logitsT.reshape([-1, this.vocabSize]);
      const flatTargets = targets.reshape([-1]);

      // convert targets to one hot vectors to conform to tf.softmaxCrossEntropy
      const oneHotTargets = tf.oneHot(flatTargets, this.vocabSize);

      // calculate and return loss
      const loss = tf.losses.softmaxCrossEntropy(oneHotTargets, flatLogits);
      return loss;
  }

  generate(context, maxTokens){
    for(let i = 0; i < maxTokens; i++){
      context = tf.tidy(() => {
        // crop context to the last block size tokens
        const start = Math.max(context.shape[1] - this.blockSize, 0);
        const sliceSize = Math.min(context.shape[1], this.blockSize);
        const croppedContext = context.slice([0, start], [-1, sliceSize]); 

        // get predictions
        const logits = this.apply(croppedContext);

        // get last time step
        const last = tf.gather(logits, logits.shape[1] - 1, 1);

        // sample from distribution
        const next = tf.multinomial(last, 1);

        // append to running sequence
        return tf.concat([context, next], 1);
      });
    }

    return context;
  } 

  getClassName() { return 'GPTLanguageModel'; }
}


// ------------------------- TRAINING LOOP -----------------------------


// define model and optimizer
const gptmodel = new GPTLanguageModel();
const optimizer = tf.train.adam(LEARNING_RATE);
gptmodel.build();

// training loop
for(let i = 0; i < MAX_ITERS; i++){
  // get batch
  const batch = getBatch("train");
  const xb = batch.x;
  const yb = batch.y;
  
  // get loss
  optimizer.minimize(() => {
    const loss = gptmodel.loss(xb, yb);
    if(i % EVAL_ITERS == 0) { loss.print(); }
    return loss;
  });

  xb.dispose();
  yb.dispose();
}

optimizer.dispose();

// decode and print results
const cont = tf.zeros([1, 1], "int32");
const batcharr = gptmodel.generate(cont, 500).arraySync()[0];
console.log(decode(batcharr));
