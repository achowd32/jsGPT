import { train } from './train.js'

const form = document.getElementById("hyperparam-form");
form.addEventListener("submit", (event) => {
  event.preventDefault();
  const batchSizeVal = form.elements['batch-size'].value;
  const blockSizeVal = form.elements['block-size'].value;
  const maxItersVal = form.elements['max-iters'].value;
  const learningRateVal = form.elements['learning-rate'].value;
  const hyperparams = {
    batchSize: parseInt(batchSizeVal),
    blockSize: parseInt(blockSizeVal),
    maxIters: parseInt(maxItersVal),
    learningRate: parseFloat(learningRateVal)
  };
  train(hyperparams);
});
