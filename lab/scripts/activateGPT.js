import { train } from './train.js'
import { clear } from './display.js'

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

  // output training notes
  const notesDisplay = document.getElementById("notes-display");
  notesDisplay.innerHTML += "<p> Started training! </p>";

  // train and output
  train(hyperparams, "nanogpt");
});
