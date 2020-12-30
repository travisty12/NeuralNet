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

  // Returns nothing -- just updates the weights and biases based on the error reported by backpropagation
  // Inputs: mini_batch, a small collection of training input/output pairs, and eta, the coefficient of learning (how radically the weights and biases should be changed with respect to the error)
  update_mini_batch(mini_batch, eta) {
    let gradient_b = TensorMath.zeroes(this.biases);
    let gradient_w = TensorMath.zeroes(this.weights);
    mini_batch.forEach(([x,y]) => {
      // Finds the error for each bias, and every weight, for each training input
      let [delta_gradient_b, delta_gradient_w] = this.backprop(x,y);
      // Sums the errors all together to find the average error for each bias and weight from all inputs in a mini batch
      gradient_b = gradient_b.map((el, i) => TensorMath.sum(el, delta_gradient_b[i]));
      gradient_w = gradient_w.map((el, i) => TensorMath.sum(el, delta_gradient_w[i]));
    });
    // Updates the weights and biases using the average error. The greater the 'eta' value, the greater the change for a given error value.
    // We multiply the gradient by a _negative_ coefficient, because the gradient represents how much an increase to 'el' would _increase_ the cost -- so we always go in the opposite direction of the gradient.
    this.weights = this.weights.map((el, i) => TensorMath.sum(el, TensorMath.product(-(eta/mini_batch.length), gradient_w[i])));
    this.biases = this.biases.map((el, i) => TensorMath.sum(el, TensorMath.product(-(eta/mini_batch.length), gradient_b[i])));
  }

  // Takes in training input and output arrays, and returns the cost gradient (i.e. error) with respect to biases and weights
  backprop(x, y) {
    let gradient_b = TensorMath.zeroes(this.biases);
    let gradient_w = TensorMath.zeroes(this.weights);
    // initial activation (just the input layer's values)
    let activation = x;
    // Array of activations for each layer
    let activations = [x];
    // Array of weighted inputs for each layer after the first
    let zs = [];
    this.biases.forEach((b,i) => {
      // Calculates and stores the weighted input and activation array for each layer
      let z = TensorMath.sum(TensorMath.dot(this.weights[i], activation), b);
      zs.push(z);
      activation = TensorMath.sigmoid(z);
      activations.push(activation);
    });

    // Delta is initially the error of the output layer. The basic Quadratic Cost function causes delta to be the cost derivative times sigmoid prime. The Cross-Entropy Cost function causes delta to just be the cost derivative.
    let delta = this.cost_derivative(activations[activations.length-1], y); // Cross Entropy
    // let delta = TensorMath.product(this.cost_derivative(activations[activations.length-1], y), TensorMath.sigmoid_prime(zs[zs.length-1])); // Quadratic

    gradient_b[gradient_b.length-1] = delta;
    gradient_w[gradient_w.length-1] = TensorMath.dot(delta, TensorMath.transpose(activations[activations.length-2]));

    // Backpropagation loop! Goes backwards through the layers, finding the errors for each layer using the errors of the layer before it.
    for (let l = 2; l < this.num_layers; l++) {
      // let z = zs[zs.length - l]; // Used for Quadratic Cost
      // let sp = TensorMath.sigmoid_prime(z); // Used for Quadratic Cost

      // The layer's weights are transposed, so instead of representing the weights _entering_ a neuron in layer l+1, they represent the weights _leaving_ the neuron in layer l. Multiplying that by the delta gives the error from each neuron in layer l, letting you travel backwards through the net.

      // delta = TensorMath.product(TensorMath.dot(TensorMath.transpose(this.weights[this.weights.length-l+1]), delta), sp); // Quadratic Cost
      delta = TensorMath.dot(TensorMath.transpose(this.weights[this.weights.length-l+1]), delta); // Cross-Entropy Cost

      gradient_b[gradient_b.length-l] = delta;
      gradient_w[gradient_w.length-l] = TensorMath.dot(delta, TensorMath.transpose(activations[activations.length-l-1]));
    }
    return [gradient_b, gradient_w];
  }

  cost_derivative(output_activations, y) {
    return output_activations.map((_, i) => output_activations[i] - y[i]);
  }

  // Using the test data fed into SGD, shows each epoch how many test inputs it can correctly identify. Just returns a number!
  evaluate(test_data) {
    // For all test data, runs it through the current network and rounds the final result to the nearest number, to compare to the theoretical expected output
    let test_results = test_data.map(([x,y]) => {
      return [this.feedForward(x).map((el) => Math.round(el)),y];
    });
    // Looping over each pair of test/theory output arrays in test_results, increments the 'sum' value if the arrays are equal. Returns the number of correct results
    return test_results.reduce((sum, [test, theory], i) => (TensorMath.arrayEquality(test, theory) ? sum + 1 : sum), 0);
  }
}


let net = new Network([10,8,4],"decimalToBinary");
let a = [
  [[1,0,0,0,0,0,0,0,0,0],[0,0,0,0]],
  [[0,1,0,0,0,0,0,0,0,0],[1,0,0,0]],
  [[0,0,1,0,0,0,0,0,0,0],[0,1,0,0]],
  [[0,0,0,1,0,0,0,0,0,0],[1,1,0,0]],
  [[0,0,0,0,1,0,0,0,0,0],[0,0,1,0]],
  [[0,0,0,0,0,1,0,0,0,0],[1,0,1,0]],
  [[0,0,0,0,0,0,1,0,0,0],[0,1,1,0]],
  [[0,0,0,0,0,0,0,1,0,0],[1,1,1,0]],
  [[0,0,0,0,0,0,0,0,1,0],[0,0,0,1]],
  [[0,0,0,0,0,0,0,0,0,1],[1,0,0,1]],
];
net.SGD(a,10000,10,0.1,a);
debugger;