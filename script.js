const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

let drawing = false;

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

// 🔥 loader with error reporting
async function loadFlat(file) {
  try {
    let res = await fetch(file);

    if (!res.ok) {
      throw new Error(`HTTP error ${res.status} for ${file}`);
    }

    let text = await res.text();

    return text
      .replace(/\r/g, "")
      .split("\n")
      .map(x => parseFloat(x.trim()))
      .filter(x => !isNaN(x));

  } catch (err) {
    document.getElementById("status").innerText =
      "Error loading " + file + ": " + err.message;
    throw err;
  }
}

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
    document.getElementById("status").innerText = "Loading files...";

    let w1_flat = await loadFlat("w1.txt");
    let b1_flat = await loadFlat("b1.txt");
    let w2_flat = await loadFlat("w2.txt");
    let b2_flat = await loadFlat("b2.txt");

    document.getElementById("status").innerText = "Reshaping...";

    w1 = reshape(w1_flat, 128, 784);
    b1 = b1_flat;
    w2 = reshape(w2_flat, 10, 128);
    b2 = b2_flat;

    document.getElementById("status").innerText = "Model loaded ✅";
    document.getElementById("result").innerText = "Draw and click Predict";
    document.getElementById("predictBtn").disabled = false;

  } catch (err) {
    console.error(err);
  }
}

loadWeights();

// activations
function relu(arr) {
  return arr.map(x => Math.max(0, x));
}

function softmax(arr) {
  let max = Math.max(...arr);
  let exps = arr.map(x => Math.exp(x - max));
  let sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

function matVecMul(matrix, vec) {
  return matrix.map(row =>
    row.reduce((sum, val, i) => sum + val * vec[i], 0)
  );
}

function forward(x) {
  let z1 = matVecMul(w1, x).map((v, i) => v + b1[i]);
  let a1 = relu(z1);
  let z2 = matVecMul(w2, a1).map((v, i) => v + b2[i]);
  return softmax(z2);
}

function getInput() {
  let tempCanvas = document.createElement("canvas");
  tempCanvas.width = 28;
  tempCanvas.height = 28;

  let tempCtx = tempCanvas.getContext("2d");
  tempCtx.drawImage(canvas, 0, 0, 28, 28);

  let data = tempCtx.getImageData(0, 0, 28, 28).data;

  let input = [];
  for (let i = 0; i < data.length; i += 4) {
    input.push(1 - data[i] / 255);
  }

  return input;
}

function predict() {
  let input = getInput();
  let out = forward(input);

  let max = Math.max(...out);
  let pred = out.indexOf(max);

  document.getElementById("result").innerText =
    "Prediction: " + pred;
}
