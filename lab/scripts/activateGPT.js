import { train } from './train.js'
import { clear, displayNotes } from './display.js'
/* TODO:
ADD BACKGROUND CHECK
ADD WEB WORKERS TO KEEP PAGE RESPONSIVE
ADD ABILITY TO CONTINUE TRAINING MODEL
UPDATE FAQ
ADD RESPONSIVE DESIGN
ADD ABILITY TO CANCEL TRAINING RUN
PREVENT USERS FROM RUNNING SEVERAL MODELS
*/

const state = {model: null};

function updateStyle() {
  const downloadBtn = document.getElementById("download-button");
  if (state.model == null){
    downloadBtn.style.cursor = "default";
    downloadBtn.style.background = "grey";
  } else {
    downloadBtn.style.cursor = "pointer";
    downloadBtn.style.background = "black";
  } 
}

// handle GPT form
const gptForm = document.getElementById("gpt-form");
gptForm.addEventListener("submit", async (event) => {
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

  // clear output and reset state
  clear();
  state.model = null;
  updateStyle();

  // output training notes
  let note = "Started training! Please note that the page may be less responsive during the training process."
  displayNotes(note);

  // train and output
  state.model = await train(hyperparams, "nanogpt");

  // update style
  updateStyle();
});

// handle download button
document.getElementById("download-button").addEventListener("click", async () => {
  // if no model exists, don't do anything
  if(state.model == null){ return; }

  // download
  await state.model.save("myModel");
});
