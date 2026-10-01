import createDigitNetwork from './network.mjs';

let network;
try {
  network = await createDigitNetwork({print: () => {}});
  if (network._initialize_model() !== 1) throw new Error('Could not load saved network');
  self.postMessage({type: 'ready'});
} catch (error) {
  self.postMessage({type: 'error', error: error.message});
}
self.onmessage = ({data: {id, pixels}}) => {
  try {
    if (!network) throw new Error('Model is not ready');
    if (!Array.isArray(pixels) || pixels.length !== 784 || pixels.some(p => !Number.isFinite(p) || p < 0 || p > 255))
      throw new Error('Expected 784 grayscale pixels from 0 to 255');
    const input = network._input_buffer();
    new Float64Array(network.HEAPU8.buffer, input, 784).set(pixels);
    const prediction = network._predict_digit();
    const probabilities = Array.from(new Float64Array(network.HEAPU8.buffer, network._output_buffer(), 10));
    if (prediction < 0 || probabilities.some(p => !Number.isFinite(p))) throw new Error('Invalid model output');
    self.postMessage({id, prediction, probabilities});
  } catch (error) {
    self.postMessage({id, error: error.message});
  }
};
