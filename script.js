const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// Fill background white initially
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

let drawing = false;

// Smooth drawing
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

// 🔹 Load flat file (one value per line)
async function loadFlat(file) {
  let res = await fetch(file);
  let text = await res.text();
  return text.trim().split(/\s+/).map(Number);
}

// 🔹 Reshape flat → matrix
function reshape(flat, rows, cols) {
  let matrix = [];
  for (let i = 0; i < rows; i++) {
    matrix.push(flat.slice(i * cols, (i + 1) * cols));
  }
  return matrix;
}

let w1, b1, w2, b2;

async function loadWeights() {
  let w1_flat = await loadFlat("w1.txt");
  let b1_flat = await loadFlat("b1.txt");
  let w2_flat = await loadFlat("w2.txt");
  let b2_flat = await loadFlat("b2.txt");

  // ✅ reshape according to architecture
  w1 = reshape(w1_flat, 128, 784);
  b1 = b1_flat;

  w2 = reshape(w2_flat, 10, 128);
  b2 = b2_flat;

  console.log("Weights loaded");
}

loadWeights();

// 🔹 Activation functions
function relu(arr) {
  return arr.map(x => Math.max(0, x));
}

function softmax(arr) {
  let max = Math.max(...arr);
  let exps = arr.map(x => Math.exp(x - max));
  let sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

// 🔹 Matrix-vector multiply
function matVecMul(matrix, vec) {
  return matrix.map(row =>
    row.reduce((sum, val, i) => sum + val * vec[i], 0)
  );
}

// 🔹 Forward pass
function forward(x) {
  let z1 = matVecMul(w1, x).map((v, i) => v + b1[i]);
  let a1 = relu(z1);

  let z2 = matVecMul(w2, a1).map((v, i) => v + b2[i]);
  let output = softmax(z2);

  return output;
}

// 🔹 Convert canvas → 28x28 input
function getInput() {
  let tempCanvas = document.createElement("canvas");
  tempCanvas.width = 28;
  tempCanvas.height = 28;

  let tempCtx = tempCanvas.getContext("2d");
  tempCtx.drawImage(canvas, 0, 0, 28, 28);

  let imgData = tempCtx.getImageData(0, 0, 28, 28).data;

  let input = [];

  for (let i = 0; i < imgData.length; i += 4) {
    let pixel = imgData[i]; // red channel
    input.push(1 - pixel / 255); // 🔥 invert for MNIST
  }

  return input;
}

// 🔹 Predict
function predict() {
  if (!w1) {
    alert("Weights still loading, please wait...");
    return;
  }

  let input = getInput();
  let output = forward(input);

  let prediction = output.indexOf(Math.max(...output));

  document.getElementById("result").innerText =
    "Prediction: " + prediction;
}
