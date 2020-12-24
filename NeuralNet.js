import TensorMath from "./TensorMath.js";

class Network {
  constructor(sizes, name) { // takes in an array whose indices represent the number of neurons in each layer, e.g. [10, 16, 4] could represent a network with 10 inputs, a hidden layer of 16 neurons, and an output layer of 4 neurons
    this.name = name,
    this.num_layers = sizes.length,
    this.sizes = sizes,
    this.biases = this.generateBiases(), // randomized biases for neurons of each layer EXCEPT the inputs on a normal distribution (gaussian curve). So for the above example, [new Array(16).fill(normal distribution), new Array(4).fill(normal distribution)]
    this.weights = this.generateWeights()// randomized weights connecting neurons of each layer on a normal distribution (gaussian curve). So for the above example, [[new Array(16).fill(new Array(10).fill(normal distrubtion))], new Array(4).fill(new Array(16).fill(normal distribution))]
  }

  generateBiases() {
    const output = [];
    this.sizes.slice(1).forEach((i) => output.push(TensorMath.randn(i)));
    return output;
  }

  generateWeights() {
    return 0;
  }
}


// let net = new Network([5,4,2],"asdf");
// console.log(net.biases);