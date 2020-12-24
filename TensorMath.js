class TensorMath {
  static normalDistribution() { // gaussian curve using Box-Muller transform (https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform)
    let [u, v] = [Math.random(), Math.random()];
    while (!(u && v)) [u, v] = [Math.random(), Math.random()]; // prevents the slim chance of either of them picking 0
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  static randn(x = 0, y = 0) { // with no inputs, returns a random normal distribution float. with 1 input, an array of length x, of normal distribution floats. with 2 inputs, it retuns a 2-D x by y array of normal distribution floats. built this way to mirror NumPy's randn function.
    const output = []; // Double input -- array of arrays.
    for (let i = 0; i < x; i++) { // can't use Array.fill, because that will force every index to have the same random value
      output.push(TensorMath.randn(y));
    }
    return output;
  }
}