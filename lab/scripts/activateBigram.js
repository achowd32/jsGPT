import { train } from './train.js'
import { clear, displayNotes, updateStyle } from './display.js'

const state = {model: null, training: false};

// handle form
const bigramForm = document.getElementById("bigram-form");
bigramForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.training){ return; }

  // update state and display
  state.training = true;
  state.model = null;
  clear();
  updateStyle(state);

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

  // output training notes
  let note = "Started training! Please note that the page may be less responsive during the training process."
  displayNotes(note);

  // train and output
  state.model = await train(hyperparams, "bigram");

  // update style
  state.training = false;
  updateStyle(state);
});

// handle download button
document.getElementById("download-button").addEventListener("click", async () => {
  // if no model exists, don't do anything
  if(state.model == null){ return; }

  // download
  await state.model.save("myModel");
});
