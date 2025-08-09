#!/usr/bin/env node

import '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';

// hyperparameters
const BATCH_SIZE = 32;
const BLOCK_SIZE = 32;
const MAX_ITERS = 10000;

// read in data file
const dataStr = fs.readFileSync('../data.txt').toString();

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

// define bigram model
class BigramLanguageModel extends tf.layers.Layer {
  constructor(vocabSize){
    super({});
    this.tokenEmbeddingTable = null;
    this.vocabSize = vocabSize;
  }

  build(){
    this.tokenEmbeddingTable = tf.layers.embedding({
      inputDim: this.vocabSize,
      outputDim: this.vocabSize,
    });
    this.tokenEmbeddingTable.build([null, BLOCK_SIZE]);
    super.build();
  }

  call(inputs){
    // get raw logits tensor from embedding table
    const logits = this.tokenEmbeddingTable.apply(inputs);
    return logits;
  }
  
  loss(inputs, targets){
    const loss = tf.tidy(() => {
      // get logits
      const logitsT = this.apply(inputs);

      // flatten logits and targets
      const flatLogits = logitsT.reshape([-1, this.vocabSize]);
      const flatTargets = targets.reshape([-1]);

      // convert targets to one hot vectors to conform to tf.softmaxCrossEntropy
      const oneHotTargets = tf.oneHot(flatTargets, this.vocabSize);

      // calculate and return loss
      return tf.losses.softmaxCrossEntropy(oneHotTargets, flatLogits);
    });
    return loss;
  }

  generate(context, maxTokens){
    const output = tf.tidy(() => {
      for(let i = 0; i < maxTokens; i++){
        // get predictions
        const logits = this.apply(context);

        // get last time step
        const last = tf.gather(logits, logits.shape[1] - 1, 1);

        // otherwise evens the resulting probabilities out and gives poor output
        const scaledLast = last.mul(tf.scalar(3)); 

        // sample from distribution
        const next = tf.multinomial(scaledLast, 1);

        // append to running sequence
        const concatLayer = tf.layers.concatenate();
        context = concatLayer.apply([context, next]);
      }
      return context;
    });

    return output;
  } 

  getClassName() { return 'BigramLanguageModel'; }
}
// define model and optimizer
const bgmodel = new BigramLanguageModel(vocabSizeVal);
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
    const lossArr = loss.arraySync();
    if(i % 1000 == 0) {console.log(`Loss at iteration ${i}: ${lossArr}`);}
    return loss;
  });

  xb.dispose();
  yb.dispose();
}

optimizer.dispose();

// decode and print results
const cont = tf.zeros([1, 1], "int32");
const batcharr = bgmodel.generate(cont, 500).arraySync()[0];
console.log(decode(batcharr));
