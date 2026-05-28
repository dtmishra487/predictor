let w1_flat, b1, w2_flat, b2;

// Load flat file (one number per line)
async function loadFlat(file) {
  try {
    console.log("Loading:", file);

    let res = await fetch(file);

    if (!res.ok) {
      throw new Error(file + " not found");
    }

    let text = await res.text();

    console.log(file + " loaded, length:", text.length);

    let arr = text.trim().split(/\s+/).map(Number);

    console.log(file + " parsed, values:", arr.length);

    return arr;

  } catch (err) {
    document.getElementById("prediction").innerText = "Error loading " + file;
    console.error(err);
    throw err;
  }
}

async function loadWeights() {
  document.getElementById("prediction").innerText = "Loading model...";

  w1_flat = await loadFlat("w1.txt");
  b1 = await loadFlat("b1.txt");
  w2_flat = await loadFlat("w2.txt");
  b2 = await loadFlat("b2.txt");

  document.getElementById("prediction").innerText = "Model loaded!";
}

loadWeights();

// Drawing
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");

ctx.fillStyle = "black";
ctx.fillRect(0, 0, 280, 280);

let drawing = false;

canvas.onmousedown = () => drawing = true;
canvas.onmouseup = () => drawing = false;

canvas.onmousemove = (e) => {
  if (!drawing) return;

  ctx.fillStyle = "white";
  ctx.beginPath();
  ctx.arc(e.offsetX, e.offsetY, 10, 0, 2 * Math.PI);
  ctx.fill();
};

function clearCanvas() {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, 280, 280);
}

// Forward pass WITHOUT reshape
function forward(x) {
  let a1 = new Array(128).fill(0);

  for (let i = 0; i < 128; i++) {
    let sum = b1[i];
    for (let j = 0; j < 784; j++) {
      sum += w1_flat[i * 784 + j] * x[j];
    }
    a1[i] = Math.max(0, sum);
  }

  let z2 = new Array(10).fill(0);

  for (let i = 0; i < 10; i++) {
    let sum = b2[i];
    for (let j = 0; j < 128; j++) {
      sum += w2_flat[i * 128 + j] * a1[j];
    }
    z2[i] = sum;
  }

  // Softmax
  let max = Math.max(...z2);
  let exps = z2.map(v => Math.exp(v - max));
  let total = exps.reduce((a, b) => a + b, 0);

  return exps.map(v => v / total);
}

function predict() {
  let img = ctx.getImageData(0, 0, 28, 28);
  let input = [];

  for (let i = 0; i < img.data.length; i += 4) {
    input.push(1 - img.data[i] / 255);
  }

  let output = forward(input);
  let pred = output.indexOf(Math.max(...output));

  document.getElementById("prediction").innerText = "Prediction: " + pred;
}
