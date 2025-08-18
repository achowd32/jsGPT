// ------------------- LAYER DEFINITIONS ------------------------

// layer to perform the scaling operation in Head
class ScaleLayer extends tf.layers.Layer {
  constructor(config) {
    super(config);
    this.scale = config.scale;
  }
  call(inputs) {
    const x = Array.isArray(inputs) ? inputs[0] : inputs;
    return tf.mul(x, tf.scalar(this.scale));
  }
  getConfig() {
    return Object.assign(super.getConfig(), {scale: this.scale});
  }
  static get className() { return 'ScaleLayer'; }
}
tf.serialization.registerClass(ScaleLayer);

// layer to perform the masking operation in Head
class CausalMask extends tf.layers.Layer {
  constructor(config) {
    super(config);
    this.blockSize = config.blockSize;
  }

  build(inputShape) {
    this.tril = tf.linalg.bandPart(tf.ones([this.blockSize, this.blockSize], 'bool'), -1, 0);  // shape [Tmax, Tmax]
    super.build(inputShape);
  }

  call(inputs) {
    const scores = Array.isArray(inputs) ? inputs[0] : inputs; // [B, T, T]
    const shape = scores.shape; // e.g. [1, 10, 10]
    const T = shape[1]; // 10 at runtime

    // slice out the [T, T] submatrix
    const mask2d = this.tril.slice([0, 0], [T, T]); // [T, T]
    const mask = mask2d.logicalNot().expandDims(0); // [1, T, T]

    const negInf = tf.fill(shape, Number.NEGATIVE_INFINITY); // [1, T, T]
    return tf.where(mask, negInf, scores); // [1, T, T]
  }

  getConfig() {
    return Object.assign(super.getConfig(), {blockSize: this.blockSize});
  }

  static get className() { return 'CausalMask'; }
}
tf.serialization.registerClass(CausalMask);

class PositionalEmbedding extends tf.layers.Layer {
  constructor(config) {
    super(config);
    this.blockSize = config.blockSize;
    this.nEmbd = config.nEmbd;
    this.embInit = tf.initializers.randomNormal({ mean: 0, stddev: 0.02 });
  }

  build(inputShape) {
    // create weights with addWeight so TFJS can save and load properly
    this.posTable = this.addWeight(
      'pos_table',
      [this.blockSize, this.nEmbd],
      'float32',
      this.embInit,
      true,
    );
    super.build(inputShape);
  }

  call(inputs) {
    // prepare inputs
    const x = Array.isArray(inputs) ? inputs[0] : inputs; // (B, T, C)
    const T = x.shape[1];

    // slice from table, add, and return
    const table = this.posTable.read(); // (BLOCK_SIZE, C)
    const posEmbd = table.slice([0, 0], [T, -1]).expandDims(0); // (1, T, C)
    return x.add(posEmbd);
  }

  getConfig() {
    return Object.assign(super.getConfig(), {
      blockSize: this.blockSize,
      nEmbd: this.nEmbd,
    });
  }
  static get className() { return 'PositionalEmbedding'; }
}
tf.serialization.registerClass(PositionalEmbedding);

// ------------------- CLASS DEFINITION ------------------------

// define GPT language model
class GPTLanguageModel {
  constructor(vocabSizeVal, hyperparams){
    // initialize vocab size property
    this.vocabSize = vocabSizeVal;

    // initialize hyperparameter properties
    this.blockSize = hyperparams.blockSize;
    this.nEmbd = hyperparams.nEmbd;
    this.nLayer = hyperparams.nLayer;
    this.nHead = hyperparams.nHead;
    this.headSize = hyperparams.headSize;
    this.dropout = hyperparams.dropout;

    // create model
    this.gptModel = this.#createGPT();
  }

  apply(inputs){
    // delegate to the internal model
    return this.gptModel.apply(inputs);
  }
  
  loss(inputs, targets){
    const returnLoss = tf.tidy(() => { 
      // get logits
      const logitsT = this.apply(inputs);

      // flatten logits and targets
      const flatLogits = logitsT.reshape([-1, this.vocabSize]);
      const flatTargets = targets.reshape([-1]);

      // convert targets to one hot vectors to conform to tf.softmaxCrossEntropy
      const oneHotTargets = tf.oneHot(flatTargets, this.vocabSize);

      // calculate and return loss
      return tf.losses.softmaxCrossEntropy(oneHotTargets, flatLogits);
    });
    return returnLoss;
  }

  generate(context, maxTokens){
    for(let i = 0; i < maxTokens; i++){
      context = tf.tidy(() => {
        // crop context to the last block size tokens
        const start = Math.max(context.shape[1] - this.blockSize, 0);
        const sliceSize = Math.min(context.shape[1], this.blockSize);
        const croppedContext = context.slice([0, start], [-1, sliceSize]); 

        // get predictions
        const logits = this.apply(croppedContext);

        // get last time step
        const last = tf.gather(logits, logits.shape[1] - 1, 1);

        // sample from distribution
        const next = tf.multinomial(last, 2).squeeze().gather([1]).expandDims(0);

        // append to running sequence
        return tf.concat([context, next], 1);
      });
    }

    return context;
  } 

  async save(filepath){
    // save the model to a file
    return this.gptModel.save(`downloads://${filepath}`);
  }

  async load(filepath){
    // load the model from a file
    this.gptModel = await tf.loadLayersModel(`file://${filepath}/model.json`);
  }

  // define function that returns a LayerNorm layer as a tf.model
  #createLayerNorm() {
    const input = tf.input({shape: [null, this.nEmbd]});
    const normalized = tf.layers.layerNormalization().apply(input);
    return tf.model({inputs: input, outputs: normalized});
  }

  // define function that returns a FeedForward layer as a tf.model
  #createFeedForward() {
    const input = tf.input({shape: [null, this.nEmbd]});
    
    // expansion layer with ReLU activation
    const expanded = tf.layers.dense({
      units: 4 * this.nEmbd,
      activation: 'relu',
    }).apply(input);
    
    // compression layer
    const compressed = tf.layers.dense({
      units: this.nEmbd,
    }).apply(expanded);
    
    // dropout layer
    const output = tf.layers.dropout({rate: this.dropout}).apply(compressed);
    return tf.model({inputs: input, outputs: output});
  }

  // define function that returns a Head layer as a tf.model
  #createHead() {
    // create symbolic input
    const input = tf.input({shape: [null, this.nEmbd], dtype: 'float32'});

    // create key, query, value embeddings; (B, T, headSize)
    const key   = tf.layers.dense({units: this.headSize, useBias: false}).apply(input);
    const query = tf.layers.dense({units: this.headSize, useBias: false}).apply(input);
    const value = tf.layers.dense({units: this.headSize, useBias: false}).apply(input);

    // compute self attention scores
    const keyT = tf.layers.permute({dims: [2, 1]}).apply(key); // (B, headSize, T)
    const scores = tf.layers.dot({axes: [2, 1]}).apply([query, keyT]); // (B, T, headSize) @ (B, headSize, T) = (B, T, T)
    const scaled = new ScaleLayer({scale: 1/Math.sqrt(this.headSize)}).apply(scores); // scale by 1/sqrt(headSize)
    
    // apply mask
    const masked = new CausalMask({blockSize: this.blockSize}).apply(scaled);
    const weights = tf.layers.activation({activation: 'softmax'}).apply(masked);

    // apply dropout
    const dropped = tf.layers.dropout({rate: this.dropout}).apply(weights);

    // perform weighted aggregation of the values
    const out = tf.layers.dot({axes: [2, 1]}).apply([dropped, value]);

    // build the model
    return tf.model({inputs: input, outputs: out});
  }

  // function that returns MultiHeadAttention as a tf.model
  #createMultiHeadAttention() {
    const input = tf.input({shape: [null, this.nEmbd]});

    // instantiate heads
    const heads = Array.from({length: this.nHead},
      () => this.#createHead());

    // apply each head in parallel
    let out = heads.map(head => head.apply(input));

    // concat each headOut value along the feature axis 
    out = tf.layers.concatenate({axis: 2}).apply(out);

    // apply projection layer
    out = tf.layers.dense({units: this.nEmbd}).apply(out);

    // apply dropout
    out = tf.layers.dropout({rate: this.dropout}).apply(out);

    return tf.model({inputs: input, outputs: out});
  }

  // function that returns a Transformer Block as a tf.model
  #createBlock() {
    const input = tf.input({shape: [null, this.nEmbd]});
    const headSize = Math.floor(this.nEmbd / this.nHead);

    // create self attention layer
    const sa = this.#createMultiHeadAttention();

    // create feed forward layer
    const ffwd = this.#createFeedForward();

    // perform computations with residual
    const ln1Out = tf.layers.layerNormalization().apply(input);
    const saOut = sa.apply(ln1Out);
    const residual1 = tf.layers.add().apply([input, saOut]); // input + sa(ln1(input))
    
    const ln2Out = tf.layers.layerNormalization().apply(residual1);
    const ffwdOut = ffwd.apply(ln2Out);
    const residual2 = tf.layers.add().apply([residual1, ffwdOut]); // residual1 + ffwd(ln2(residual1))

    return tf.model({inputs: input, outputs: residual2});
  }

  // function that returns a complete GPT Language Model as a tf.model
  #createGPT() {
    const input = tf.input({shape: [this.blockSize], dtype: 'int32'});

    // token embedding table
    const tokEmbd = tf.layers.embedding({
      inputDim: this.vocabSize,
      outputDim: this.nEmbd,
    }).apply(input); // (B, T, N_EMBD)

    // positional embedding table
    const posEmbdLayer = new PositionalEmbedding(
      {blockSize: this.blockSize, nEmbd: this.nEmbd});
    const posEmbd = posEmbdLayer.apply(tokEmbd);

    // apply all transformer blocks sequentially
    let blockEmbd = posEmbd;
    for(let i = 0; i < this.nLayer; i++){
      const block = this.#createBlock();
      blockEmbd = block.apply(blockEmbd);
    }

    // final layer normalization
    const normalized = tf.layers.layerNormalization().apply(blockEmbd);

    // linear layer to vocabulary size
    const logits = tf.layers.dense({
      units: this.vocabSize,
    }).apply(normalized); // (B, T, vocabSize)

    return tf.model({inputs: input, outputs: logits});
  }

  getClassName() { return 'GPTLanguageModel'; }
}

export { GPTLanguageModel };
