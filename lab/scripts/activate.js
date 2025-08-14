const form = document.getElementById("hyperparam-form");
form.addEventListener("submit", (event) => {
  event.preventDefault();
  const batchVal = form.elements['batch-size'].value;
  const blockVal = form.elements['block-size'].value;
  document.getElementById("batch-size-res").innerHTML = batchVal;
  document.getElementById("block-size-res").innerHTML = blockVal;
});
