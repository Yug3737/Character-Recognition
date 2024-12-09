// file: letter_recognition.js
// author: Yug Patel
// last modified: 5 Dec 2024

const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
const clearButton = document.getElementById('clearButton');
const saveButton = document.getElementById('saveButton');

// Making canvas background as white
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);

let drawing = false;

// start drawing
canvas.addEventListener('mousedown', (e) => {
    drawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
});

// Draw
canvas.addEventListener('mousemove', (e) => {
    if (drawing) {
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 30;
        ctx.lineCap = 'round';
        ctx.stroke();
    }
});

// stop drawing
canvas.addEventListener('mouseup', () => drawing = false);
canvas.addEventListener('mouseout', () => drawing = false);

// Clear Canvas
clearButton.addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStlye = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
});

saveButton.addEventListener('click', () => {
    let canvas = document.getElementById("drawingCanvas");
    let imageData = canvas.toDataURL('image/png');
    fetch('/save_letter_image', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: imageData }),
    })
        .then(res => res.json())
        .then(data => {
            // display top 3 predictions
            const predictions = data.predictions;
            const predictionsList = document.getElementById("predictions");

            // clear prev results
            predictionsList.innerHTML = "";

            predictions.forEach(prediction => {
                const listItem = document.createElement("li");
                listItem.textContent = `Letter: ${prediction.predicted_letter}, Probability: ${prediction.probability}`;
                predictionsList.appendChild(listItem);
            });
        })
        .catch(err => {
            console.log("Error: ", err);
        })
});