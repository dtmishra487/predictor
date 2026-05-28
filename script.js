const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// White background
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

let drawing = false;

// Drawing events
canvas.addEventListener("mousedown", () => drawing = true);
canvas.addEventListener("mouseup", () => drawing = false);
canvas.addEventListener("mouseleave", () => drawing = false);
canvas.addEventListener("mousemove", draw);

function draw(e) {
  if (!drawing) return;

  ctx.fillStyle = "black";
  ctx.beginPath();
  ctx.arc(e.offsetX, e.offsetY, 10, 0, Math.PI * 2);
  ctx.fill();
}

function clearCanvas() {
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  document.getElementById("result").innerText = "";
}

// 🔥 Robust loader (newline-safe)
async function loadFlat(file) {
  let res = await fetch("./" + file);
  let text = await res.text();

  return text
    .replace(/\r/g, "")
    .split("\n")
    .map(x => parseFloat(x.trim()))
    .filter(x => !isNaN(x));
}

// reshape flat → matrix
function reshape(flat, rows, cols) {
  if (flat.length !== rows * cols) {
    throw new Error(`Shape mismatch: expected ${rows*cols}, got ${flat.length}`);
  }

  let matrix = [];
  for (let i = 0; i < rows; i++) {
    matrix.push(flat.slice(i * cols, (i + 1) * cols));
  }
  return matrix;
}

let w1, b1, w2, b2;

async function loadWeights() {
  try {
    let w1_flat = await loadFlat("w1.txt");
    let b1_flat = await loadFlat("b1.txt");
    let w2_flat = await loadFlat("w2.txt");
    let b2_flat = await loadFlat("b2.txt");

    // reshape
    w1 = reshape(w1_flat, 128, 784);
    b1 = b1_flat;

    w2 = reshape(w2_flat, 10, 128);
    b2 = b2_flat;

    console.log("Model loaded successfully");

    document.getElementById("predictBtn").disabled = false;
    document.getElementById("result").innerText = "Draw a digit and click Predict";

  } catch (err) {
    console.error(err);
    document.getElementById("result").innerText = "Error loading model!";
  }
}

loadWeights();

// Activation
function relu(arr) {
  return arr.map(x => Math.max(0, x));
}

function softmax(arr) {
  let max = Math.max(...arr);
  let exps = arr.map(x => Math.exp(x - max));
  let sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

// Matrix-vector multiply
function matVecMul(matrix, vec) {
  return matrix.map(row =>
    row.reduce((sum, val, i) => sum + val * vec[i], 0)
  );
}

// Forward pass
function forward(x) {
  let z1 = matVecMul(w1, x).map((v, i) => v + b1[i]);
  let a1 = relu(z1);

  let z2 = matVecMul(w2, a1).map((v, i) => v + b2[i]);
  let output = softmax(z2);

  return output;
}

// Canvas → input
function getInput() {
  let tempCanvas = document.createElement("canvas");
  tempCanvas.width = 28;
  tempCanvas.height = 28;

  let tempCtx = tempCanvas.getContext("2d");
  tempCtx.drawImage(canvas, 0, 0, 28, 28);

  let imgData = tempCtx.getImageData(0, 0, 28, 28).data;

  let input = [];

  for (let i = 0; i < imgData.length; i += 4) {
    let pixel = imgData[i];
    input.push(1 - pixel / 255); // MNIST inversion
  }

  return input;
}

// Predict
function predict() {
  if (!w1) {
    alert("Model still loading...");
    return;
  }

  let input = getInput();
  let output = forward(input);

  console.log("Output:", output);

  let maxVal = Math.max(...output);
  let prediction = output.indexOf(maxVal);

  if (prediction === -1 || isNaN(maxVal)) {
    document.getElementById("result").innerText = "Prediction failed (NaN issue)";
  } else {
    document.getElementById("result").innerText = "Prediction: " + prediction;
  }
}
