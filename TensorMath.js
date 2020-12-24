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
}