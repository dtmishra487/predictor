let w1, b1, w2, b2;

// load file
async function load(file) {
  let res = await fetch(file);
  let txt = await res.text();
  return txt.trim().split(/\s+/).map(Number);
}

async function init() {
  try {
    w1 = await load("w1.txt");
    b1 = await load("b1.txt");
    w2 = await load("w2.txt");
    b2 = await load("b2.txt");

    document.getElementById("result").innerText = "Model loaded!";
    document.getElementById("predictBtn").disabled = false;
  } catch (e) {
    document.getElementById("result").innerText = "Error loading model";
    console.error(e);
  }
}

init();

// canvas
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");

ctx.fillStyle = "black";
ctx.fillRect(0,0,280,280);

let draw = false;
canvas.onmousedown = ()=>draw=true;
canvas.onmouseup = ()=>draw=false;

canvas.onmousemove = (e)=>{
  if(!draw) return;
  ctx.fillStyle="white";
  ctx.beginPath();
  ctx.arc(e.offsetX,e.offsetY,10,0,2*Math.PI);
  ctx.fill();
};

function clearCanvas(){
  ctx.fillStyle="black";
  ctx.fillRect(0,0,280,280);
}

// forward pass (NO reshape)
function forward(x){
  let a1 = new Array(128).fill(0);

  for(let i=0;i<128;i++){
    let sum = b1[i];
    for(let j=0;j<784;j++){
      sum += w1[i*784 + j] * x[j];
    }
    a1[i] = Math.max(0,sum);
  }

  let out = new Array(10).fill(0);

  for(let i=0;i<10;i++){
    let sum = b2[i];
    for(let j=0;j<128;j++){
      sum += w2[i*128 + j] * a1[j];
    }
    out[i]=sum;
  }
console.log("Output:", out);
  return out.indexOf(Math.max(...out));
}

function predict(){
  let img = ctx.getImageData(0,0,28,28);
  let x=[];

  for(let i=0;i<img.data.length;i+=4){
    x.push(1 - img.data[i]/255);
  }

  let p = forward(x);
  console.log("Sample input:", x.slice(0,10));
  document.getElementById("result").innerText = "Prediction: "+p;
}
