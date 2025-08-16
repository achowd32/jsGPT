import { dataLoader } from './dataLoader.js'
import { BigramLanguageModel } from '../src/modelBigram.js'
import { GPTLanguageModel } from '../src/modelGPT.js'
import { displayLoss, displaySample } from './display.js'

function initModel(dl, hyperparams, modelType){
  if (modelType === "bigram"){
    return new BigramLanguageModel(dl.vocabSize, hyperparams);
  } else if (modelType === "nanogpt"){
    return new GPTLanguageModel(dl.vocabSize, hyperparams);
  }
}

async function estimateLoss(dl, model){
  const out = {};
  for(const split of ['train', 'val']){
    let sum = 0;
    for (let k = 0; k < 20; k++) {
      const { x, y } = dl.getBatch(split);
      const lossT = model.loss(x, y);
      const lossNum = (await lossT.data())[0];
      lossT.dispose(); x.dispose(); y.dispose();

      sum += lossNum;
    }
    out[split] = sum / 20;
  }
  return out;
}

async function trainLoop(dl, model, optimizer, hyperparams){
  // setup hyperparameters
  const learningRate = hyperparams.learningRate;
  const maxIters = hyperparams.maxIters;
  const evalInterval = hyperparams.evalInterval;

  // training loop
  for(let i = 0; i < maxIters; i++){
    // output progress 
    if (i % evalInterval == 0){
      await tf.nextFrame();
      const trainValLoss = await estimateLoss(dl, model);
      displayLoss(i, trainValLoss["train"].toFixed(4), trainValLoss["val"].toFixed(4));
    }
      
    tf.tidy(() => {
      // get batch
      const batch = dl.getBatch("train");
      const xb = batch.x; const yb = batch.y;
      
      // get loss
      optimizer.minimize(() => { return model.loss(xb, yb); });
    });
  }

  // output loss after the final training iteration
  await tf.nextFrame();
  const trainValLoss = await estimateLoss(dl, model);
  displayLoss(maxIters, trainValLoss["train"].toFixed(4), trainValLoss["val"].toFixed(4));
}

async function train(hyperparams, modelType){
  // setup backend; TODO: add try/catch for failures
  await tf.setBackend("webgpu");

  // read in data file and setup dataloader object
  const dataStr = await (await fetch('./data.txt')).text();
  const dl = new dataLoader(dataStr, hyperparams);

  // define model and optimizer
  const model = initModel(dl, hyperparams, modelType);
  const optimizer = tf.train.adam(hyperparams.learningRate);

  // initiate training loop
  await trainLoop(dl, model, optimizer, hyperparams);

  // decode and print results
  const cont = tf.zeros([1, 1], "int32");
  const gen = model.generate(cont, 200);
  const batcharr = await gen.array();
  displaySample(dl.decode(batcharr[0]));
  
  // dispose
  optimizer.dispose();
  cont.dispose();
  gen.dispose();
}

export { train };
