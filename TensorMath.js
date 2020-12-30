class TensorMath {

  // gaussian curve using Box-Muller transform (https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform)
  static normalDistribution() { 
    let [u, v] = [Math.random(), Math.random()];
    // prevents the slim chance of either of them picking 0
    while (!(u && v)) [u, v] = [Math.random(), Math.random()]; 
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  // With no inputs, returns a random normal distribution float. With 1 input, an array of length x, of normal distribution floats. With 2 inputs, it retuns a 2-D x by y array of normal distribution floats. Built this way to mirror NumPy's randn function.
  static randn(x = 0, y = 0) { 
    // Base case: just return normal distribution
    if (!(x || y)) return TensorMath.normalDistribution();

    // Single input -- array of length x of base case returned
    if (!y) { 
      const output = []; 
      for (let i = 0; i < x; i++) { // can't use Array.fill, because that will force every index to have the same random value
        output.push(TensorMath.randn());
      }
      return output;
    }

    // Double input -- array of arrays.
    const output = []; 
    for (let i = 0; i < x; i++) { // can't use Array.fill, because that will force every index to have the same random value
      output.push(TensorMath.randn(y));
    }
    return output;
  }

  // given an array, returns an array of the same shape, with 0 for values. Used to initialize deltas for weights and biases
  static zeroes(input) {
    return input.map((el) => (typeof(el) == 'number' ? 0 : TensorMath.zeroes(el)));
  }

  // Transposes an array. ex. [ [1,2], [3,4], [5,6] ] ==> [ [1,2,3], [4,5,6] ]
  static transpose(input) {
    if (typeof(input[0]) == 'number') return input;
    return input[0].map((_,i) => input.map((_,j) => input[j][i]))
  }

  // Activation function, 1 / (1 + e^(-z)). Always a value between 0 and 1
  // If input is an array, it returns an array of activations
  static sigmoid(z) {
    if (typeof(z) == 'object') return z.map((el) => TensorMath.sigmoid(el));
    return 1.0 / (1.0 + Math.exp(-z));
  }

  // Derivative of the activation, conveniently equal to sigmoid * (1 - sigmoid)
  static sigmoid_prime(z) {
    if (typeof(z) == 'object') return z.map((el) => TensorMath.sigmoid_prime(el));
    return TensorMath.product(TensorMath.sigmoid(z), (1 - TensorMath.sigmoid(z)));
  }

  // Takes in 2 arrays, returns a bool on whether they represent the same info
  static arrayEquality(u,v) {
    if (u === v) return true;
    if (u == null || v == null) return false;
    if (u.length != v.length) return false;

    for (let i = 0; i < u.length; i++) {
      if (u[i] != v[i]) return false;
    }
    return true;
  }

  // Adds two arrays (or an array and a scalar, or two scalars, depending on type of inputs)
  static sum(u,v) {
    const uScalar = typeof(u) == 'number';
    const vScalar = typeof(v) == 'number';
    if (uScalar && vScalar) return u + v;
    if (uScalar) return v.map((el) => TensorMath.sum(u, el));
    if (vScalar) return u.map((el) => TensorMath.sum(v, el));
    return u.map((el,i) => TensorMath.sum(el,v[i]))
  }  
  
  // Multiplies two arrays of the same exact size (or an array and a scalar, or two scalars, depending on type of inputs)
  static product(u,v) {
    const uScalar = typeof(u) == 'number';
    const vScalar = typeof(v) == 'number';
    if (uScalar && vScalar) return u * v;
    if (uScalar) return v.map((el) => TensorMath.product(u, el));
    if (vScalar) return u.map((el) => TensorMath.product(v, el));
    return u.map((el,i) => TensorMath.product(el,v[i]))
  }

  // Gives the dot product of two arrays-of-arrays
  static dot(u,v) {
    return u.map((el,i) => {
      if (el.length == v.length) {
        // Case: used to find weighted inputs (before adding bias into account), in feedforward and backprop. Also used when multiplying a layer's weights by the delta in backprop
        return el.reduce((summand, _, j) => summand + el[j] * v[j], 0);
      } else {
        // Case: in backprop. Dot product of delta and the transposed activations of a layer.
        return v.map(el2 => el2 * el);
      }
    });
  }
}

export default TensorMath;