function clear(){
  const notesDiv = document.getElementById("notes-display");
  const lossDiv = document.getElementById("loss-display");
  const sampleDiv = document.getElementById("sample-display");
  notesDiv.innerHTML =  '';
  lossDiv.innerHTML =  '';
  sampleDiv.innerHTML =  '';
}

function displayNotes(note) {
  const notesDisplay = document.getElementById("notes-display");
  notesDisplay.innerHTML += `<p><b>${note}</b></p>`;
}

function displayLoss(i, trainLoss, valLoss){
  const lossDiv = document.getElementById("loss-display");
  lossDiv.innerHTML +=  `<p>At iteration ${i}: train loss — ${trainLoss}, validation loss — ${valLoss}. </p>`;
}

function displaySample(sample){
  const sampleDiv = document.getElementById("sample-display");
  sampleDiv.innerHTML +=  `<br><p><b>Your model generated the following text sample:</b></p>`;
  sampleDiv.innerHTML +=  `<p>${sample}</p>`;
}

function updateStyle(state) {
  const trainBtn = document.getElementById("train-button");
  const downloadBtn = document.getElementById("download-button");
  if (state.model == null){
    downloadBtn.style.cursor = "default";
    downloadBtn.style.background = "grey";
  } else {
    downloadBtn.style.cursor = "pointer";
    downloadBtn.style.background = "black";
  } 

  if (state.training){
    trainBtn.style.cursor = "default";
    trainBtn.style.background = "grey";
  } else {
    trainBtn.style.cursor = "pointer";
    trainBtn.style.background = "black";
  }
}

export { clear, displayNotes, displayLoss, displaySample, updateStyle };
