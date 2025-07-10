#!/usr/bin/env node

import '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';

// hyperparameters
const BATCH_SIZE = 32;
const BLOCK_SIZE = 8;
const MAX_ITERS = 10000;

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

// define bigram model
class BigramLanguageModel extends tf.layers.Layer {
  constructor(vocabSize){
    super({});
    this.tokenEmbeddingTable = tf.layers.embedding({
      inputDim: vocabSize,
      outputDim: vocabSize
    });
    this.vocabSize = vocabSize;
  }

  call(inputs, kwargs = {}){
    // get raw logits tensor from embedding table
    const logitsT = this.tokenEmbeddingTable.apply(inputs);

    if(Object.keys(kwargs).length == 0){
      // if no targets specified, just return logits
      return {logits: logitsT, loss: null};
    } else{
      // if targets specified, perform loss calculations
      const targets = kwargs.targets;

      // flatten logits and targets
      const flatLogits = logitsT.reshape([-1, this.vocabSize]);
      const flatTargets = targets.reshape([-1]);

      // convert targets to one hot vectors to conform to tf.softmaxCrossEntropy
      const oneHotTargets = tf.oneHot(flatTargets, this.vocabSize);

      // calculate and return loss, along with logits
      const lossT = tf.losses.softmaxCrossEntropy(oneHotTargets, flatLogits);
      return {logits: logitsT, loss: lossT};
    }
  }
  
  generate(context, maxTokens){
    for(let i = 0; i < maxTokens; i++){
      // get predictions
      const logits = this.apply(context).logits;

      // get last time step
      const last = tf.gather(logits, logits.shape[1] - 1, 1);

      // perform softmax normalization
      const probs = tf.softmax(last);

      // sample from distribution
      const next = tf.multinomial(probs, 1);

      // append to running sequence
      const concatLayer = tf.layers.concatenate();
      context = concatLayer.apply([context, next]);
    }
    
    return context;
  } 

  getClassName() { return 'BigramLanguageModel'; }
}
// define model and optimizer
const bgmodel = new BigramLanguageModel(vocabSizeVal);
const optimizer = tf.train.adam(0.001);

tf.tidy(() => {
  // a single forward pass on a zero tensor of the right shape
  bgmodel.call(tf.zeros([1, BLOCK_SIZE], 'int32'), {});
});

// training loop
for(let i = 0; i < MAX_ITERS; i++){
  const batch = getBatch("train");
  const xb = batch.x;
  const yb = batch.y;
  tf.tidy(() => {
    const lossScalar = optimizer.minimize(() => {
      // pull out the vector loss and reduce it
      const { loss: lossVec } = bgmodel.call(xb, { targets: yb });
      return lossVec.mean();    // <-- scalar
    }, /* returnCost */ true);

    // log every so often
    if (i % 500 === 0) {
      console.log(`Iteration ${i}: loss = ${lossScalar.dataSync()[0].toFixed(4)}`);
    }

    // dispose the scalar loss
    lossScalar.dispose();
  }); 
}

// decode and print results
const cont = tf.zeros([1, 1], "int32");
const batcharr = bgmodel.generate(cont, 200).arraySync()[0];
console.log(decode(batcharr));




