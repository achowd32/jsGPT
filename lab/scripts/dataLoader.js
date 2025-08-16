// define dataLoader class
class dataLoader {
  // fields
  charList; vocabSize; encoder; decoder; dataTensor; trainTensor; valTensor;

  // constructor
  constructor(dataStr, hyper){
    // initialize hyperparameters
    this.batchSize = hyper.batchSize;
    this.blockSize = hyper.blockSize;

    this.#setTokenizers(dataStr); // setup token encoder and decoder
    this.#setData(dataStr); // setup training and validation data
  }

  #setTokenizers(dataStr) {
    // get array of all characters and set vocab size
    this.charList = Array.from(new Set(dataStr)).sort();
    this.vocabSize = this.charList.length;
    
    // define encoder and decoder maps using character list
    this.encoder = new Map(this.charList.map((val, ind) => [val, ind]));
    this.decoder = new Map(this.charList.map((val, ind) => [ind, val]));
  }

  #setData(dataStr) {
    // convert all data into tensor by encoding
    this.dataTensor = tf.tensor(this.encode(dataStr), undefined, "int32");

    // split data into train and validation tensors
    const trainSize = Math.round(0.9 * this.dataTensor.size);
    const valSize = this.dataTensor.size - trainSize;
    this.trainTensor = this.dataTensor.slice([0], [trainSize]);
    this.valTensor = this.dataTensor.slice([trainSize], [valSize]);
  }

  encode(str){
    return Array.from(str).map(c => this.encoder.get(c));
  }

  decode(toks){
    return toks.map(i => this.decoder.get(i)).join('');
  }

  getBatch(split){
    return tf.tidy(() => {
      // establish which data to use
      let data = this.trainTensor;
      if(split === "val"){
        data = this.valTensor;
      }
      
      // indices to sample from
      let minInd = 0;
      let maxInd = data.size - this.blockSize - 1; 
      const randInds = tf.randomUniform([this.batchSize], minInd, maxInd, "int32").arraySync();

      // get samples
      const xRows = [];
      const yRows = [];
      for(let i = 0; i < this.batchSize; i++){
        let curSplit = randInds[i];
        let xTensor = data.slice([curSplit], [this.blockSize]);
        let yTensor = data.slice([curSplit + 1], [this.blockSize]);
        xRows.push(xTensor);
        yRows.push(yTensor);
      }
      
      // use stack to convert to 2D tensor
      const xVal = tf.stack(xRows);
      const yVal = tf.stack(yRows);
      return { x: tf.keep(xVal), y: tf.keep(yVal) };
    });
  }
}

export { dataLoader };
