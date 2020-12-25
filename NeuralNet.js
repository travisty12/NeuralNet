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
}


// let net = new Network([5,4,2],"asdf");
// console.log(net.weights);