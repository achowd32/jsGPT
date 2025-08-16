import { train } from './train.js'

function clear(){
  const lossDiv = document.getElementById("loss-display");
  const sampleDiv = document.getElementById("sample-display");
  lossDiv.innerHTML =  '';
  sampleDiv.innerHTML =  '';
}

function displayLoss(i, trainLoss, valLoss){
  const lossDiv = document.getElementById("loss-display");
  lossDiv.innerHTML +=  `<p>At iteration ${i}: train loss — ${trainLoss}, validation loss — ${valLoss}. </p>`;
}

function displaySample(sample){
  const sampleDiv = document.getElementById("sample-display");
  sampleDiv.innerHTML +=  `<p> Your model generated the following text sample: </p>`;
  sampleDiv.innerHTML +=  `<p> ${sample} </p>`;
}

// handle bigram form
//const bigramForm = document.getElementById("bigram-form");
/*bigramForm.addEventListener("submit", (event) => {
  event.preventDefault();

  // get hyperparameter values
  const batchSizeVal = bigramForm.elements['batch-size'].value;
  const blockSizeVal = bigramForm.elements['block-size'].value;
  const learningRateVal = bigramForm.elements['learning-rate'].value;
  const evalItvlVal = bigramForm.elements['eval-itvl'].value;
  const maxItersVal = bigramForm.elements['max-iters'].value;

  // define hyperparameter object
  const hyperparams = {
    batchSize: parseInt(batchSizeVal),
    blockSize: parseInt(blockSizeVal),
    learningRate: parseFloat(learningRateVal),
    evalInterval: parseInt(evalItvlVal),
    maxIters: parseInt(maxItersVal),
  };

  // clear output divs
  clear();

  // train and output
  train(hyperparams, displayLoss, displaySample);
});*/

// handle GPT form
const gptForm = document.getElementById("gpt-form");
gptForm.addEventListener("submit", (event) => {
  event.preventDefault();

  // get hyperparameter values
  const batchSizeVal = gptForm.elements['batch-size'].value;
  const blockSizeVal = gptForm.elements['block-size'].value;
  const learningRateVal = gptForm.elements['learning-rate'].value;
  const evalItvlVal = gptForm.elements['eval-itvl'].value;
  const nEmbdVal = gptForm.elements['n-embd'].value;
  const nLayerVal = gptForm.elements['n-layer'].value;
  const nHeadVal = gptForm.elements['n-head'].value;
  const headSizeVal = gptForm.elements['head-size'].value;
  const dropoutVal = gptForm.elements['dropout'].value;
  const maxItersVal = gptForm.elements['max-iters'].value;

  // define hyperparameter object
  const hyperparams = {
    batchSize: parseInt(batchSizeVal),
    blockSize: parseInt(blockSizeVal),
    learningRate: parseFloat(learningRateVal),
    evalInterval: parseInt(evalItvlVal),
    nEmbd: parseInt(nEmbdVal),
    nLayer: parseInt(nLayerVal),
    nHead: parseInt(nHeadVal),
    headSize: parseInt(headSizeVal),
    dropout: parseFloat(dropoutVal),
    maxIters: parseInt(maxItersVal),
  };

  // clear output divs
  clear();

  // train and output
  train(hyperparams, displayLoss, displaySample);
});
