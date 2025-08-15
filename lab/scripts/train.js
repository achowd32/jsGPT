import { BigramLanguageModel } from '../models/modelBigram.js'
import { BATCH_SIZE, BLOCK_SIZE, MAX_ITERS } from '../models/modelBigram.js'

// setup backend; TODO: add try/catch for failures
await tf.setBackend("webgpu");

// read in data file
const dataStr = await (await fetch('./data.txt')).text();

// TODO: setup data loader object; takes dataStr, initializes charList, stoi, itos
// has methods encode, decode, and getBatch
// set up token encoder and decoder
const charList = Array.from(new Set(dataStr)).sort();
const vocabSizeVal = charList.length;
const stoi = new Map(charList.map((val, ind) => [val, ind]));
const itos = new Map(charList.map((val, ind) => [ind, val]));
let encode = function(str){ return Array.from(str).map(c => stoi.get(c)); }
let decode = function(toks){ return toks.map(i => itos.get(i)).join(''); }

// set up training and validation data
const dataTensor = tf.tensor(encode(dataStr), undefined, "int32");
const trainSize = Math.round(0.9 * dataTensor.size);
const valSize = dataTensor.size - trainSize;

const trainTensor = dataTensor.slice([0], [trainSize]);
const valTensor = dataTensor.slice([trainSize], [valSize]);

// set up data loader
function getBatch(split){
  return tf.tidy(() => {
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
    return { x: tf.keep(xVal), y: tf.keep(yVal) };
  });
}

// define model and optimizer
const bgmodel = new BigramLanguageModel(vocabSizeVal);
const optimizer = tf.train.adam(0.0001);

// training loop
for(let i = 0; i < MAX_ITERS; i++){
  tf.tidy(() => {
    // get batch
    const batch = getBatch("train");
    const xb = batch.x;
    const yb = batch.y;
    
    // output progress 
    if (i % 1000 == 0){ console.log(i); }
    
    // get loss
    optimizer.minimize(() => { return bgmodel.loss(xb, yb); });
  });
}
 
// dispose of the optimizer
optimizer.dispose();

// decode and print results
const cont = tf.zeros([1, 1], "int32");
const gen = bgmodel.generate(cont, 200);
const batcharr = await gen.array();
console.log(decode(batcharr[0]));
cont.dispose();
gen.dispose();
