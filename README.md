# Neural Net

#### Two classes meant to aid in quickly building a neural network of any size and depth.

### Travis Scott

### Installation / Setup

* Run `git clone https://github.com/travisty12/NeuralNet` from the console to download, `cd NeuralNet` to enter the directory
* To create a new net in the file, create a new instance of `Network`, with the size of the layers and a name as arguments, define training data, and run the SGD method with training data, number of epochs, batch size, learning coefficient, and test data as arguments. See the bottom of `NeuralNet.js` for an example.
* To run from console and interact with the net after initialization, uncomment the `debugger` from the bottom of `NeuralNet.js`, and run `node inspect NeuralNet.js`. Run `cont` until it hits the `debugger` line and run `repl`, at which point you can interact with the object, i.e. `console.log(net.feedForward(dummyData));`

### Other

* All concepts used in this program were learned from Michael Nielsen's book *Neural Networks and Deep Learning*, found [here](http://neuralnetworksanddeeplearning.com/). This repo is just a translation of stochastic gradient descent concepts shown in the book in Python, into JavaScript, along with a second class built to handle tensor math (previously handled by NumPy)
* I understand that TensorFlow.js exists for this, but I just thought it would be a fun challenge to write the math and network creation myself! 