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

// ------------------- MODEL DEFINITIONS ------------------------

// define function that returns a LayerNorm layer as a tf.model
function createLayerNorm() {
  const input = tf.input({shape: [null, N_EMBD]});
  const normalized = tf.layers.layerNormalization().apply(input);
  return tf.model({inputs: input, outputs: normalized});
}

// define function that returns a Head layer as a tf.model
function createHead(headSize) {
  // create symbolic input
  const input = tf.input({shape: [null, N_EMBD], dtype: 'float32'});

  // create key, query, value embeddings; (B, T, headSize)
  const key   = tf.layers.dense({units: headSize, useBias: false}).apply(input);
  const query = tf.layers.dense({units: headSize, useBias: false}).apply(input);
  const value = tf.layers.dense({units: headSize, useBias: false}).apply(input);

  // compute self attention scores
  const keyT = tf.layers.permute({dims: [2, 1]}).apply(key); // (B, headSize, T)
  const scores = tf.layers.dot({axes: [2, 1]}).apply([query, keyT]); // (B, T, headSize) @ (B, headSize, T) = (B, T, T)
  const scaled = new ScaleLayer({scale: 1/Math.sqrt(headSize)}).apply(scores); // scale by 1/sqrt(headSize)
  
  // apply mask
  const masked = new CausalMask({blockSize: BLOCK_SIZE}).apply(scaled);
  const weights = tf.layers.activation({activation: 'softmax'}).apply(masked);

  // apply dropout
  const dropped = tf.layers.dropout({rate: DROPOUT}).apply(weights);

  // perform weighted aggregation of the values
  const out = tf.layers.dot({axes: [2, 1]}).apply([dropped, value]);

  // build the model
  return tf.model({inputs: input, outputs: out});
}

// define function that returns a FeedForward layer as a tf.model
function createFeedForward() {
  const input = tf.input({shape: [null, N_EMBD]});
  
  // expansion layer with ReLU activation
  const expanded = tf.layers.dense({
    units: 4 * N_EMBD,
    activation: 'relu',
  }).apply(input);
  
  // compression layer
  const compressed = tf.layers.dense({
    units: N_EMBD,
  }).apply(expanded);
  
  // dropout layer
  const output = tf.layers.dropout({rate: DROPOUT}).apply(compressed);
  return tf.model({inputs: input, outputs: output});
}

// function that returns MultiHeadAttention as a tf.model
function createMultiHeadAttention(numHeads, headSize) {
  const input = tf.input({shape: [null, N_EMBD]});

  // instantiate heads
  const heads = Array.from({length: numHeads},
    () => createHead(headSize));

  // apply each head in parallel
  let out = heads.map(head => head.apply(input));

  // concat each headOut value along the feature axis 
  out = tf.layers.concatenate({axis: 2}).apply(out);

  // apply projection layer
  out = tf.layers.dense({units: N_EMBD}).apply(out);

  // apply dropout
  out = tf.layers.dropout({rate: DROPOUT}).apply(out);

  return tf.model({inputs: input, outputs: out});
}

// function that returns a Transformer Block as a tf.model
function createBlock(nHead) {
  const input = tf.input({shape: [null, N_EMBD]});
  const headSize = Math.floor(N_EMBD / nHead);

  // create self attention layer
  const sa = createMultiHeadAttention(nHead, headSize);

  // create feed forward layer
  const ffwd = createFeedForward();

  // create layerNorm layers, or use the Identity layer to avoid layerNorm
  const ln1 = createLayerNorm();
  const ln2 = createLayerNorm();

  // perform computations with residual
  const ln1Out = ln1.apply(input);
  const saOut = sa.apply(ln1Out);
  const residual1 = tf.layers.add().apply([input, saOut]); // input + sa(ln1(input))
  
  const ln2Out = ln2.apply(residual1);
  const ffwdOut = ffwd.apply(ln2Out);
  const residual2 = tf.layers.add().apply([residual1, ffwdOut]); // residual1 + ffwd(ln2(residual1))

  return tf.model({inputs: input, outputs: residual2});
}

// function that returns a complete GPT Language Model as a tf.model (without position embeddings)
function createGPT() {
  const input = tf.input({shape: [BLOCK_SIZE], dtype: 'int32'});

  // token embedding table
  const tokEmbd = tf.layers.embedding({
    inputDim: vocabSizeVal,
    outputDim: N_EMBD,
  }).apply(input); // (B, T, N_EMBD)

  // positional embedding table
  const posEmbdLayer = new PositionalEmbedding({blockSize: BLOCK_SIZE, nEmbd: N_EMBD});
  const posEmbd = posEmbdLayer.apply(tokEmbd);

  // apply all transformer blocks sequentially
  let blockEmbd = posEmbd;
  for(let i = 0; i < N_LAYER; i++){
    const block = createBlock(N_HEAD);
    blockEmbd = block.apply(blockEmbd);
  }

  // final layer normalization
  const normalized = tf.layers.layerNormalization().apply(blockEmbd);

  // linear layer to vocabulary size
  const logits = tf.layers.dense({
    units: vocabSizeVal,
  }).apply(normalized); // (B, T, vocabSize)

  return tf.model({inputs: input, outputs: logits});
}

// define GPT language model
class GPTLanguageModel {
  constructor(vocabSizeVal, hyperparams){
    this.vocabSize = vocabSizeVal;
    this.blockSize = BLOCK_SIZE;
    this.gptModel = createGPT();
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
        const next = tf.multinomial(last, 1);

        // append to running sequence
        return tf.concat([context, next], 1);
      });
    }

    return context;
  } 

  async save(filepath){
    // save the model to a file
    return this.gptModel.save(`file://${filepath}`);
  }

  async load(filepath){
    // load the model from a file
    this.gptModel = await tf.loadLayersModel(`file://${filepath}/model.json`);
  }

  getClassName() { return 'GPTLanguageModel'; }
}

export { GPTLanguageModel };
