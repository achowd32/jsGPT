function clear(){
  const notesDiv = document.getElementById("notes-display");
  const lossDiv = document.getElementById("loss-display");
  const sampleDiv = document.getElementById("sample-display");
  notesDiv.innerHTML =  '';
  lossDiv.innerHTML =  '';
  sampleDiv.innerHTML =  '';
}

function displayNotes() {
  const notesDisplay = document.getElementById("notes-display");
  notesDisplay.innerHTML += "<p><b>Started training! Please note that the page may be less responsive during the training process.</b></p>";
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

export { clear, displayNotes, displayLoss, displaySample };
