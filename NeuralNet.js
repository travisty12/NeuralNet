class Network {
  constructor(sizes, name) { // takes in an array whose indices represent the number of neurons in each layer, e.g. [10, 16, 4] could represent a network with 10 inputs, a hidden layer of 16 neurons, and an output layer of 4 neurons
    this.name = name,
    this.num_layers = sizes.length,
    this.sizes = sizes
  }
}