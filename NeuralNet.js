import TensorMath from "./TensorMath.js";

class Network {
  constructor(sizes, name) { // takes in an array whose indices represent the number of neurons in each layer, e.g. [5, 4, 2] could represent a network with 10 inputs, a hidden layer of 16 neurons, and an output layer of 4 neurons
    this.name = name,
    this.num_layers = sizes.length,
    this.sizes = sizes,
    this.biases = this.generateBiases(), // randomized biases for neurons of each layer EXCEPT the inputs on a normal distribution (gaussian curve). So for the above example, [new Array(4).fill(normal distribution), new Array(2).fill(normal distribution)]
    this.weights = this.generateWeights() // randomized weights connecting neurons of each layer on a normal distribution (gaussian curve). So for the above example, [[new Array(4).fill(new Array(5).fill(normal distrubtion))], new Array(2).fill(new Array(4).fill(normal distribution))]
  }

  generateBiases() {
    // Returns an array, each index being the size of each layer AFTER the inputs, randomized values for initial biases.
    return this.sizes.slice(1).map((i) => TensorMath.randn(i)); 
  }

  generateWeights() {
    const pairs = [];
    for (let i = 1; i < this.num_layers; i++) {
      pairs.push([this.sizes[i-1],this.sizes[i]]);
    }
    return pairs.map((i) => TensorMath.randn(i[1],i[0]));
  }

  // Takes in a set of initial inputs, and runs them through each layer, returning the output layer's activations
  // Each layer, the dot product of the weights and input returns an array with the shape of the next layer's neurons. They get summed with the biases to get the weighted inputs for the next layer, and the sigmoid function returns their activation. That activation is then used as the input for the next loop's dot product until the final layer.
  feedForward(a) {
    this.weights.forEach((_,i) => {
      a = TensorMath.sigmoid(TensorMath.sum(TensorMath.dot(this.weights[i], a), this.biases[i]));
    });
    return a;
  }

  // Stochastic Gradient Descent -- the main method that needs to be called after construction to train the weights and biases
  SGD(training_data, epochs, mini_batch_size, eta, test_data = null) {
    let n_test;
    if (test_data) n_test = test_data.length;
    const n = training_data.length;

    // loops over all training data for a set number of epochs
    for (let i = 0; i < epochs; i++) {
      training_data.sort(() => Math.random() - 0.5);
      const mini_batches = [];

      // splits training data into mini batches
      for (let j = 0; j < n; j += mini_batch_size) {
        mini_batches.push(training_data.slice(j, j+mini_batch_size));
      }
      let progress = 0;
      // updates the weights and biases for each mini batch
      mini_batches.forEach((mini_batch, k) => {
        this.update_mini_batch(mini_batch, eta);
        // logs the progress through each epoch (most useful in larger networks)
        if (Math.floor((100.0 * k * mini_batch_size) / n) != progress) {
          progress = Math.floor((100.0 * k * mini_batch_size) / n);
          console.log(`${progress}% through epoch ${i}`);
        }
      });
      if (test_data) {
        console.log(`Epoch ${i}: ${this.evaluate(test_data)} / ${n_test}`);
      } else {
        console.log(`Epoch ${i} complete.`);
      }
    }
  }

  update_mini_batch(mini_batch, eta) {

  }

  evaluate(test_data) {

  }
}


let net = new Network([5,4,2],"asdf");
console.log(net.weights);