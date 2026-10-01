'use strict';
// C++ runs in a worker so inference never blocks pointer events.
const modelWorker = new Worker('./model-worker.js', {type: 'module'});
const pendingPredictions = new Map();
let nextPredictionId = 0;
let modelLoaded = false;
let modelError = null;
let resolveModel, rejectModel;
const modelReady = new Promise((resolve, reject) => {resolveModel = resolve; rejectModel = reject;});
modelWorker.onmessage = ({data}) => {
  if (data.type === 'ready') {modelLoaded = true; resolveModel(); return;}
  if (data.type === 'error') {failModel(new Error(data.error)); return;}
  const pending = pendingPredictions.get(data.id);
  if (!pending) return;
  pendingPredictions.delete(data.id);
  if (data.error) pending.reject(new Error(data.error));
  else pending.resolve(data);
};
function failModel(error) {
  modelError = error;
  rejectModel(error);
  for (const pending of pendingPredictions.values()) pending.reject(error);
  pendingPredictions.clear();
}
modelWorker.onerror = () => failModel(new Error('Unable to load browser model. Reload to retry.'));
async function predictPixels(pixels) {
  await modelReady;
  if (modelError) throw modelError;
  return new Promise((resolve, reject) => {
    const id = ++nextPredictionId;
    pendingPredictions.set(id, {resolve, reject});
    modelWorker.postMessage({id, pixels});
  });
}

const drawing = document.querySelector('#drawing');
const ctx = drawing.getContext('2d', {willReadFrequently: true});
const preview = document.querySelector('#preview');
const pctx = preview.getContext('2d');
const status = document.querySelector('#status');
const bars = document.querySelector('#bars');
for (let i = 0; i < 10; i++) {
  const row = document.createElement('div');
  row.className = 'row';
  row.innerHTML = `<span>${i}</span><div class="track"><div class="fill"></div></div><span class="value">—</span>`;
  bars.append(row);
}
let activePointer = null, lastPoint = null, revision = 0, timer = null, busy = false, queued = false;

// Only input preparation happens in JavaScript. Prediction uses C++ predict().
function prepareInput() {
  const w = drawing.width, h = drawing.height;
  const rgba = ctx.getImageData(0, 0, w, h).data;
  let left = w, top = h, right = -1, bottom = -1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (rgba[(y * w + x) * 4] > 0) {
      left = Math.min(left, x); right = Math.max(right, x);
      top = Math.min(top, y); bottom = Math.max(bottom, y);
    }
  }
  if (right < 0) return null;
  const bw = right - left + 1, bh = bottom - top + 1;
  // Render at 10x resolution, then area-average for antialiased grayscale pixels.
  const scaled = document.createElement('canvas');
  scaled.width = scaled.height = 280;
  const sctx = scaled.getContext('2d');
  sctx.fillStyle = 'black'; sctx.fillRect(0, 0, 280, 280);
  const scale = 200 / Math.max(bw, bh);
  sctx.drawImage(drawing, left, top, bw, bh, (280 - bw * scale) / 2, (280 - bh * scale) / 2, bw * scale, bh * scale);
  const data = sctx.getImageData(0, 0, 280, 280).data;
  const raster = new Array(784).fill(0);
  let mass = 0, mx = 0, my = 0;
  for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
    let sum = 0;
    for (let dy = 0; dy < 10; dy++) for (let dx = 0; dx < 10; dx++)
      sum += data[((y * 10 + dy) * 280 + x * 10 + dx) * 4];
    const value = Math.round(sum / 100);
    raster[y * 28 + x] = value;
    mass += value; mx += x * value; my += y * value;
  }
  if (!mass) return null;
  // Center the ink's center of mass without clipping the bounding box.
  let minX = 28, maxX = 0, minY = 28, maxY = 0;
  raster.forEach((v, i) => {if(v) {minX=Math.min(minX,i%28);maxX=Math.max(maxX,i%28);minY=Math.min(minY,Math.floor(i/28));maxY=Math.max(maxY,Math.floor(i/28));}});
  const dx = Math.max(-minX, Math.min(27-maxX, Math.round(13.5 - mx / mass)));
  const dy = Math.max(-minY, Math.min(27-maxY, Math.round(13.5 - my / mass)));
  const pixels = new Array(784).fill(0);
  for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
    const sx = x - dx, sy = y - dy;
    if (sx >= 0 && sx < 28 && sy >= 0 && sy < 28) pixels[y*28+x] = raster[sy*28+sx];
  }
  return pixels;
}
function showInput(pixels) {
  const image = pctx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    image.data[i*4] = image.data[i*4+1] = image.data[i*4+2] = pixels?.[i] || 0;
    image.data[i*4+3] = 255;
  }
  pctx.putImageData(image, 0, 0);
}
function resetPrediction() {
  document.querySelector('#digit').textContent = '—';
  document.querySelector('#confidence').textContent = 'Waiting for ink';
  document.querySelector('#prediction-note').textContent = 'Every stroke gives it more to go on.';
  for (const row of bars.children) {
    row.classList.remove('winner'); row.querySelector('.fill').style.width = '0%'; row.querySelector('.value').textContent = '—';
  }
}
async function predict() {
  timer = null;
  if (busy) {queued = true; return;}
  const pixels = prepareInput();
  showInput(pixels);
  if (!pixels) return;
  const requestRevision = revision;
  busy = true; queued = false;
  status.textContent = 'Reading your drawing…';
  const start = performance.now();
  try {
    const result = await predictPixels(pixels);
    if (requestRevision !== revision) return;
    document.querySelector('#digit').textContent = result.prediction;
    document.querySelector('#confidence').textContent = `${(result.probabilities[result.prediction]*100).toFixed(1)}%`;
    document.querySelector('#prediction-note').textContent = `Looks like a ${result.prediction}`;
    [...bars.children].forEach((row,i) => {
      row.classList.toggle('winner', i === result.prediction);
      row.querySelector('.fill').style.width = `${result.probabilities[i]*100}%`;
      row.querySelector('.value').textContent = `${(result.probabilities[i]*100).toFixed(1)}%`;
    });
    status.textContent = `Live · ${Math.round(performance.now()-start)} ms`;
  } catch (error) {
    if (requestRevision === revision) {
      resetPrediction(); status.textContent = 'Model unavailable — reload to retry.';
      console.error('Prediction failed:', error);
    }
  } finally {
    busy = false;
    if (queued) {queued = false; schedule();}
  }
}
function schedule() {
  // Throttle (not debounce): predictions keep updating during continuous strokes.
  if (timer === null) timer = setTimeout(predict, 80);
}
function point(event) {
  const rect = drawing.getBoundingClientRect();
  // Account for CSS border and responsive scaling independently of device pixels.
  const width = drawing.clientWidth, height = drawing.clientHeight;
  return {x:(event.clientX-rect.left-drawing.clientLeft)*drawing.width/width,
          y:(event.clientY-rect.top-drawing.clientTop)*drawing.height/height};
}
function ink(p) {
  ctx.strokeStyle = ctx.fillStyle = 'white'; ctx.lineWidth = 24; ctx.lineCap = ctx.lineJoin = 'round';
  if (lastPoint) {ctx.beginPath();ctx.moveTo(lastPoint.x,lastPoint.y);ctx.lineTo(p.x,p.y);ctx.stroke();}
  else {ctx.beginPath();ctx.arc(p.x,p.y,12,0,Math.PI*2);ctx.fill();}
  lastPoint = p; revision++;
  document.querySelector('#hint').style.display = 'none';
  schedule();
}
drawing.addEventListener('pointerdown', event => {
  if (activePointer !== null || (event.pointerType === 'mouse' && event.button !== 0)) return;
  event.preventDefault();activePointer=event.pointerId;drawing.setPointerCapture(event.pointerId);lastPoint=null;ink(point(event));
});
drawing.addEventListener('pointermove', event => {
  if (event.pointerId !== activePointer) return;
  event.preventDefault();ink(point(event));
});
function endStroke(event) {
  if (event.pointerId !== activePointer) return;
  activePointer=null;lastPoint=null;schedule();
}
drawing.addEventListener('pointerup', endStroke);
drawing.addEventListener('pointercancel', endStroke);
drawing.addEventListener('lostpointercapture', endStroke);
document.querySelector('#clear').addEventListener('click', () => {
  revision++;queued=false;
  if (timer!==null) {clearTimeout(timer);timer=null;}
  if (activePointer!==null && drawing.hasPointerCapture(activePointer)) drawing.releasePointerCapture(activePointer);
  activePointer=null;lastPoint=null;
  ctx.fillStyle='black';ctx.fillRect(0,0,drawing.width,drawing.height);
  showInput(null);resetPrediction();document.querySelector('#hint').style.display='flex';status.textContent=modelError ? 'Model unavailable — reload to retry.' : modelLoaded ? 'Ready when you are' : 'Loading your model…';
});
document.querySelector('#clear').click();

modelReady.then(() => { if (!prepareInput()) status.textContent = 'Ready when you are'; }).catch(error => {status.textContent = 'Model unavailable — reload to retry.'; console.error(error);});
