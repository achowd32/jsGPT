import { train } from './train.js'

function clear(){
  const lossDiv = document.getElementById("loss-display");
  const sampleDiv = document.getElementById("sample-display");
  lossDiv.innerHTML =  '';
  sampleDiv.innerHTML =  '';
}

function displayLoss(loss){
  const lossDiv = document.getElementById("loss-display");
  lossDiv.innerHTML +=  `<p> current iteration is ${loss} </p>`;
}

function displaySample(sample){
  const sampleDiv = document.getElementById("sample-display");
  sampleDiv.innerHTML +=  `<p> Your model generated the following text sample: </p>`;
  sampleDiv.innerHTML +=  `<p> ${sample} </p>`;
}

const form = document.getElementById("hyperparam-form");
form.addEventListener("submit", (event) => {
  event.preventDefault();

  // get hyperparameter values
  const batchSizeVal = form.elements['batch-size'].value;
  const blockSizeVal = form.elements['block-size'].value;
  const maxItersVal = form.elements['max-iters'].value;
  const learningRateVal = form.elements['learning-rate'].value;

  // define hyperparameter object
  const hyperparams = {
    batchSize: parseInt(batchSizeVal),
    blockSize: parseInt(blockSizeVal),
    maxIters: parseInt(maxItersVal),
    learningRate: parseFloat(learningRateVal)
  };

  // clear output divs
  clear();

  // train and output
  train(hyperparams, displayLoss, displaySample);
});
