// hyperparameters
const BATCH_SIZE = 32;
const BLOCK_SIZE = 32;
const MAX_ITERS = 10000;

// define bigram model
class BigramLanguageModel {
  constructor(vocabSize) {
    this.vocabSize = vocabSize;
    // build the embedding layer
    this.embedding = tf.layers.embedding({
      inputDim: vocabSize,
      outputDim: vocabSize,
    });

    // model input shape: [batch, block_size]
    const input = tf.input({shape: [BLOCK_SIZE], dtype: 'int32'});
    const logits = this.embedding.apply(input);
    this.model = tf.model({inputs: input, outputs: logits});
  }

  apply(inputs) {
    // forward pass
    return this.model.apply(inputs);
  }

  loss(inputs, targets) {
    const loss = tf.tidy(() => {
      // get logits
      const logitsT = this.apply(inputs);

      // flatten logits and targets
      const flatLogits = logitsT.reshape([-1, this.vocabSize]);
      const flatTargets = targets.reshape([-1]);

      // convert targets to one hot vectors
      const oneHotTargets = tf.oneHot(flatTargets, this.vocabSize);

      // calculate and return loss
      return tf.losses.softmaxCrossEntropy(oneHotTargets, flatLogits);
    });
    return loss;
  }

  generate(context, maxTokens) {
    for (let i = 0; i < maxTokens; i++) {
      context = tf.tidy(() => {
        // get predictions
        const logits = this.apply(context);
        // get last time step
        const last = tf.gather(logits, logits.shape[1] - 1, 1);
        // scale logits
        const scaledLast = last.mul(tf.scalar(3));
        // sample from distribution
        const next = tf.multinomial(scaledLast, 2).squeeze().gather([1]).expandDims(0);
        // append to running sequence
        return tf.concat([context, next], 1);
      });
    }
    return context;
  }

  getWeights() {
    return this.model.getWeights();
  }

  save(filename){
    return this.model.save(filename);
  }

  async load(filename){
    this.model = await tf.loadLayersModel(filename);
  }
}

export { BigramLanguageModel };
export { BLOCK_SIZE, BATCH_SIZE, MAX_ITERS };
