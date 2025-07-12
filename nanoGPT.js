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
  constructor(){
    super({});
    this.vocabSize = vocabSizeVal;
    this.nEmbd = N_EMBD;
    this.headSize = HEAD_SIZE;
    this.blockSize = BLOCK_SIZE;
    this.built = false;

    const ones = tf.ones([this.blockSize, this.blockSize]);
    this.tril = tf.linalg.bandPart(ones, -1, 0);
  }

  build(){
    // key layer
    this.key = tf.layers.dense({
      inputDim: this.nEmbd,
      units: this.vocabSize, 
      useBias: false,
    });
    
    // query layer
    this.query = tf.layers.dense({
      inputDim: this.nEmbd,
      units: this.vocabSize, 
      useBias: false,
    });

    // value layer
    this.value = tf.layers.dense({
      inputDim: this.nEmbd,
      units: this.vocabSize,
      useBias: false,
    });
  }

  call(x){
    // get dimensions of input embeddings
    const [B, T, C] = x.shape;

    // pass embeddings through key and query layers
    const k = this.key.apply(x); // (B, T, C)
    const q = this.key.apply(x); // (B, T, C)
    
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

    // perform weighted aggregation of the values
    const v = this.value.apply(x); (B, T, C)
    const out = tf.matMul(wei, v);
    return out;
  }

  getClassName() { return 'Head'; }
}
const testHead = new Head();
testHead.build();

// define bigram model
class BigramLanguageModel extends tf.layers.Layer {
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
    const logits = this.lmHead.apply(embdSum); // (B, T, vocabSize)
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
      const minTimeStep = Math.max(context.shape[1] - 8, 0);
      const indSlice = tf.range(minTimeStep, context.shape[1], 1, "int32")
      const croppedContext = tf.gather(context, indSlice, 1); 

      // get predictions
      const logits = this.apply(croppedContext);

      // get last time step
      const last = tf.gather(logits, logits.shape[1] - 1, 1);

      // sample from distribution
      const next = tf.multinomial(last, 1);

      // append to running sequence
      const concatLayer = tf.layers.concatenate();
      context = concatLayer.apply([context, next]);
    }

    return context;
  } 

  getClassName() { return 'BigramLanguageModel'; }
}


// ------------------------- TRAINING LOOP -----------------------------


// define model and optimizer
const bgmodel = new BigramLanguageModel();
const optimizer = tf.train.adam(0.0001);
bgmodel.build();

// training loop
for(let i = 0; i < MAX_ITERS; i++){
  // get batch
  const batch = getBatch("train");
  const xb = batch.x;
  const yb = batch.y;
  
  // get loss
  optimizer.minimize(() => {
    const loss = bgmodel.loss(xb, yb);
    return loss;
  });

  xb.dispose();
  yb.dispose();
}

optimizer.dispose();

// decode and print results
const cont = tf.zeros([1, 1], "int32");
const batcharr = bgmodel.generate(cont, 200).arraySync()[0];
console.log(decode(batcharr));
//console.log(bgmodel.getWeights());
