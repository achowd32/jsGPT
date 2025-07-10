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

function customLoss(yTrue, yPred) {
  // Example: Mean Squared Error (MSE)
  const diff = tf.sub(yTrue, yPred);
  const squaredDiff = tf.square(diff);
  const loss = tf.mean(squaredDiff);
  return loss;
}

async function main(){
  // setup model
  const model = tf.sequential();
  model.add(tf.layers.embedding({inputDim: vocabSizeVal, outputDim: vocabSizeVal}));
  model.add(tf.layers.activation({activation: 'softmax'}));
  model.compile({loss: customLoss, optimizer: 'adam'});
  model.summary();

  // setup data
  const batch = getBatch("train");
  const xb = batch.x;
  const yb = batch.y;

  // train
  await model.fit(xb, yb, {
    epochs: 1,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(`Epoch {epoch}: loss = ${logs.loss}`);
      }
    }
  });

  // generate
  console.log("generating...");
  let cont = xb.slice([0, 0], [1, 1]);
  const maxToks = 200;

  for(let i = 0; i < maxToks; i++){
    const logits = model.predict(cont);  
    // pick out the last step correctly:
    const lastLogits = tf.gather(logits, cont.shape[1]-1, 1).squeeze([0]);  
    // choose the most likely next token:
    const nextId = lastLogits.argMax().reshape([1,1]);  
    cont = tf.concat([cont, nextId], 1);
  }
  const outToks = cont.arraySync()[0];
  console.log(decode(outToks));
};

main();
