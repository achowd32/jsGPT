#!/usr/bin/env node

import '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';

// hyperparameters
const BATCH_SIZE = 32;
const BLOCK_SIZE = 8;
const MAX_ITERS = 10000;
const N_EMBD = 32;
const HEAD_SIZE = 16;
const LEARNING_RATE = 0.001;

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
    this.built = false;

    const ones = tf.ones([this.blockSize, this.blockSize]);
    this.tril = tf.linalg.bandPart(ones, -1, 0);
  }

  build(){
    // key layer
    this.key = tf.layers.dense({
      inputDim: this.nEmbd,
      units: this.headSize, 
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
  }

  call(x){
    // get dimensions of input embeddings
    const [B, T, C] = x.shape;

    // pass embeddings through key and query layers
    const k = this.key.apply(x); // (B, T, C)
    const q = this.query.apply(x); // (B, T, C)
    
    // compute self attention scores
    const k_t = tf.transpose(k, [0, 2, 1]); // (B, C, T)
    let wei = tf.matMul(q, k_t); // (B, T, T)
    wei = wei.mul(tf.scalar(1 / Math.sqrt(C))); // scale by 1/sqrt(C)

    // create mask
    const tril = this.tril.slice([0, 0], [T, T]); // (T, T)
    const mask = tril.equal(0).expandDims(0); // (1, T, T)
    const negInf = tf.fill(wei.shape, Number.NEGATIVE_INFINITY);

    // apply mask
    wei = tf.where(mask, negInf, wei); // where mask is true, set -inf
    wei = tf.softmax(wei, /* axis */ -1);

    // perform weighted aggregation of the values
    const v = this.value.apply(x); (B, T, C)
    const out = tf.matMul(wei, v);
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
    this.built = false;

    // instantiate heads
    this.heads = Array.from({length: numHeads},
      () => new Head(this.headSize));
  }

  build(inputShape) {
    // forward the build call to each head
    this.heads.forEach(head => head.build(inputShape));
    this.built = true;
  }

  call(x) {
    // apply each head in parallel
    const headOuts = this.heads.map(head => head.apply(x));  
    // concat each headOut value along the feature axis 
    return tf.concat(headOuts, 2);
  }

  getClassName() {
    return 'MultiHeadAttention';
  }
}

class FeedForward extends tf.layers.Layer {
  constructor(nEmbd){
    super({});
    this.nEmbd = nEmbd;
  }

  build(){
    this.ff = tf.layers.dense({
      inputDim: this.nEmbd,
      units: this.nEmbd,
      activation: 'relu',
    });
  }
  
  call(inputs){
    return this.ff.apply(inputs);
  }

  getClassName(){
    return 'FeedForward';
  }
}

// define GPT language model
class GPTLanguageModel extends tf.layers.Layer {
  constructor(){
    super({});
    this.vocabSize = vocabSizeVal;
    this.nEmbd = N_EMBD;
    this.blockSize = BLOCK_SIZE;
    this.built = false;
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

    // single head of self-attention
    this.saHeads = new MultiHeadAttention(4, this.nEmbd / 4); // FIX FLOOR DIV
    this.saHeads.build();

    // feed forward layer
    this.ffwd = new FeedForward(this.nEmbd);
    this.ffwd.build();

    // build linear layer
    this.lmHead = tf.layers.dense({
      units: this.vocabSize, // output dimension
      inputDim: this.nEmbd,
    });

    this.built = true;
  }

  call(inputs){
    // get input shape
    const shape = inputs.shape;
    const B = shape[0];
    const T = shape[1];
    // get raw logits tensor from embedding tables
    const tokEmbd = this.tokenEmbeddingTable.apply(inputs); // (B, T, nEmbd)
    const posEmbd = this.positionEmbeddingTable.apply(
      tf.range(0, T, 1, "int32")); // (T, nEmbd)
    const posEmbdExpanded = posEmbd.expandDims(0); // (1, T, nEmbd)
    
    const embdSum = tokEmbd.add(posEmbdExpanded); // (B, T, nEmbd)
    const saEmbd = this.saHeads.apply(embdSum);
    const ffwdEmbd = this.ffwd.apply(saEmbd);
    const logits = this.lmHead.apply(ffwdEmbd); // (B, T, vocabSize)
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
      const start = Math.max(context.shape[1] - 8, 0);
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
    if(i % 500 == 0) { loss.print(); }
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
