import { train } from './train.js'
import { clear, displayNotes } from './display.js'

// handle form
const bigramForm = document.getElementById("bigram-form");
bigramForm.addEventListener("submit", (event) => {
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

  // output training notes
  displayNotes();

  // train and output
  train(hyperparams, "bigram");
});
