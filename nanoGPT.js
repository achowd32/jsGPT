#!/usr/bin/env node

import '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';

// hyperparameters
const BATCH_SIZE = 32;
const BLOCK_SIZE = 64;
const MAX_ITERS = 200;
const N_EMBD = 32;
const HEAD_SIZE = 16;
const LEARNING_RATE = 0.001;
const EVAL_ITERS = 50;

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


// ------------------- MODEL DEFINITIONS ------------------------
class Head extends tf.layers.Layer{
  constructor(headSize){
    super({});
    this.vocabSize = vocabSizeVal;
    this.headSize = headSize;
    this.nEmbd = N_EMBD;
    this.blockSize = BLOCK_SIZE;

    // create mask template to be applied after computing self attention scores
    const ones = tf.ones([this.blockSize, this.blockSize]);
    this.tril = tf.linalg.bandPart(ones, -1, 0);
  }

  build(){
    // key layer
    this.key = tf.layers.dense({
      inputDim: this.nEmbd,
      units: this.headSize, // 'units' are the output dimensions
      useBias: false,
    });
    
    // query layer
    this.query = tf.layers.dense({
      inputDim: this.nEmbd,
      units: this.headSize, 
      useBias: false,
    });

    // value layer
    this.value = tf.layers.dense({
      inputDim: this.nEmbd,
      units: this.headSize,
      useBias: false,
    });

    super.build();
  }

  call(x){
    // get dimensions of input embeddings
    const [B, T, C] = x.shape;

    // pass embeddings through key and query layers
    const k = this.key.apply(x); // (B, T, headSize)
    const q = this.query.apply(x); // (B, T, headSize)
    
    // compute self attention scores
    const k_t = tf.transpose(k, [0, 2, 1]); // (B, headSize, T)
    let wei = tf.matMul(q, k_t); // (B, T, headSize) @ (B, headSize, T) = (B, T, T)
    wei = wei.mul(tf.scalar(1 / Math.sqrt(this.headSize))); // scale by 1/sqrt(headSize)

    // create mask
    const tril = this.tril.slice([0, 0], [T, T]); // (T, T)
    const mask = tril.equal(0).expandDims(0); // (1, T, T)
    const negInf = tf.fill(wei.shape, Number.NEGATIVE_INFINITY);

    // apply mask
    wei = tf.where(mask, negInf, wei); // where mask is true, set -inf
    wei = tf.softmax(wei, -1); // (B, T, T) 

    // perform weighted aggregation of the values
    const v = this.value.apply(x); // (B, T, headSize)
    const out = tf.matMul(wei, v); // (B, T, T) @ (B, T, headSize) = (B, T, headSize)
    return out;
  }

  getClassName() { return 'Head'; }
}

// define MultiHeadAttention
class MultiHeadAttention extends tf.layers.Layer {
  constructor(numHeads, headSize) {
    super({});
    this.vocabSize = vocabSizeVal;
    this.numHeads = numHeads;
    this.headSize = headSize;
    this.nEmbd = N_EMBD;
    this.blockSize = BLOCK_SIZE;

    // instantiate heads
    this.heads = Array.from({length: numHeads},
      () => new Head(this.headSize));
  }

  build() {
    // forward the build call to each head
    this.heads.forEach(head => head.build());

    // projection layer
    this.proj = tf.layers.dense({
      inputDim: this.nEmbd,
      units: this.nEmbd,
    });

    super.build();
  }

  call(x) {
    // apply each head in parallel
    let out = this.heads.map(head => head.apply(x));  

    // concat each headOut value along the feature axis 
    out = tf.concat(out, 2);

    // apply projection layer
    out = this.proj.apply(out);

    return out;
  }

  getClassName() { return 'MultiHeadAttention'; }
}

// define FeedForward layer
class FeedForward extends tf.layers.Layer {
  constructor(nEmbd){
    super({});
    this.nEmbd = nEmbd;
  }

  build(){
    this.expand = tf.layers.dense({
      inputDim: this.nEmbd,
      units: 4 * this.nEmbd,
      activation: 'relu',
    });

    this.compress = tf.layers.dense({
      inputDim: 4 * this.nEmbd,
      units: this.nEmbd,
    });

    super.build();
  }
  
  call(inputs){
    let out = this.expand.apply(inputs);
    out = this.compress.apply(out);
    return out;
  }

  getClassName(){ return 'FeedForward'; }
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
    this.sa = new MultiHeadAttention(this.nHead, this.headSize);

    // create feed forward layer
    this.ffwd = new FeedForward(this.nEmbd);

    // create layerNormalization layers
    this.ln1 = tf.layers.layerNormalization();
    this.ln2 = tf.layers.layerNormalization();

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
    this.nEmbd = N_EMBD;
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
    for(let i  = 0; i < 3; i++){
      const blk = new Block(this.nEmbd, 4);
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
      context = tf.concat([context, next], 1);
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
//console.log(gptmodel.getWeights());
