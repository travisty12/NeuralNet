class TensorMath {
  static normalDistribution() { // gaussian curve using Box-Muller transform (https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform)
    let [u, v] = [Math.random(), Math.random()];
    while (!(u && v)) [u, v] = [Math.random(), Math.random()]; // prevents the slim chance of either of them picking 0
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }
}