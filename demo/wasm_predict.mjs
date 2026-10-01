// Test driver for the exact generated module shipped to browsers.
import createDigitNetwork from './static/network.mjs';
import {createInterface} from 'node:readline';
const model = await createDigitNetwork({print: () => {}});
if (model._initialize_model() !== 1) throw new Error('Checkpoint failed to load');
for await (const line of createInterface({input: process.stdin})) {
  const pixels = line.trim().split(/\s+/).map(Number);
  if (pixels.length !== 784) throw new Error('Expected 784 inputs');
  new Float64Array(model.HEAPU8.buffer, model._input_buffer(), 784).set(pixels);
  const prediction = model._predict_digit();
  const probabilities = Array.from(new Float64Array(model.HEAPU8.buffer, model._output_buffer(), 10));
  console.log(JSON.stringify({prediction, probabilities}));
}
