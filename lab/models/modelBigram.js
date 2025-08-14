// setup backend; TODO: add try/catch for failures
await tf.setBackend("webgpu");

// hyperparameters
const BATCH_SIZE = 32;
const BLOCK_SIZE = 32;
const MAX_ITERS = 5000;

// read in data file
const dataStr = await (await fetch('./data.txt')).text();

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
class BigramLanguageModel {
  constructor(vocabSize) {
    this.vocabSize = vocabSize;
    // build the embedding layer
    this.embedding = tf.layers.embedding({
      inputDim: vocabSize,
      outputDim: vocabSize,
    });

    // model input shape: [batch, block_size]
    const input = tf.input({shape: [BLOCK_SIZE], dtype: 'int32'});
    const logits = this.embedding.apply(input);
    this.model = tf.model({inputs: input, outputs: logits});
  }

  apply(inputs) {
    // forward pass
    return this.model.apply(inputs);
  }

  loss(inputs, targets) {
    const loss = tf.tidy(() => {
      // get logits
      const logitsT = this.apply(inputs);

      // flatten logits and targets
      const flatLogits = logitsT.reshape([-1, this.vocabSize]);
      const flatTargets = targets.reshape([-1]);

      // convert targets to one hot vectors
      const oneHotTargets = tf.oneHot(flatTargets, this.vocabSize);

      // calculate and return loss
      return tf.losses.softmaxCrossEntropy(oneHotTargets, flatLogits);
    });
    return loss;
  }

  generate(context, maxTokens) {
    for (let i = 0; i < maxTokens; i++) {
      context = tf.tidy(() => {
        // get predictions
        const logits = this.apply(context);
        // get last time step
        const last = tf.gather(logits, logits.shape[1] - 1, 1);
        // scale logits
        const scaledLast = last.mul(tf.scalar(3));
        // sample from distribution
        const next = tf.multinomial(scaledLast, 2).squeeze().gather([1]).expandDims(0);
        // append to running sequence
        return tf.concat([context, next], 1);
      });
    }
    return context;
  }

  getWeights() {
    return this.model.getWeights();
  }

  save(filename){
    return this.model.save(filename);
  }

  async load(filename){
    this.model = await tf.loadLayersModel(filename);
  }
}
// define model and optimizer
const bgmodel = new BigramLanguageModel(vocabSizeVal);
const optimizer = tf.train.adam(0.0001);

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
 
// dispose of the optimizer
optimizer.dispose();

// decode and print results
const cont = tf.zeros([1, 1], "int32");
const batcharr = await bgmodel.generate(cont, 200).arraySync();
console.log(decode(batcharr[0]));
