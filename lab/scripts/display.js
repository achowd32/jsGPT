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

export { clear, displayLoss, displaySample };
