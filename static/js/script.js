function predict() {
    const message = document.getElementById('message').value.trim();
    if (!message) {
        alert('Please enter a message');
        return;
    }

    // Show loading
    document.getElementById('loading').style.display = 'flex';
    document.getElementById('results').style.display = 'none';

    // Send prediction request
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `message=${encodeURIComponent(message)}`
    })
    .then(response => response.json())
    .then(data => {
        // Hide loading
        document.getElementById('loading').style.display = 'none';
        document.getElementById('results').style.display = 'block';

        // Update result box
        const resultBox = document.getElementById('result-box');
        const resultText = document.getElementById('result-text');
        if (data.prediction === 1) {
            resultBox.className = 'result-box spam';
            resultText.textContent = '⚠️ SPAM';
        } else {
            resultBox.className = 'result-box ham';
            resultText.textContent = '✅ Not Spam';
        }

        // Create and update gauge chart
        const gaugeData = {
            type: "indicator",
            mode: "gauge+number",
            value: data.confidence * 100,
            gauge: {
                axis: { range: [0, 100] },
                bar: { color: "darkblue" },
                bgcolor: "white",
                borderwidth: 2,
                bordercolor: "gray",
                steps: [
                    { range: [0, 33], color: "lightgray" },
                    { range: [33, 66], color: "gray" },
                    { range: [66, 100], color: "darkgray" }
                ],
                threshold: {
                    line: { color: "red", width: 4 },
                    thickness: 0.75,
                    value: data.confidence * 100
                }
            }
        };

        const layout = {
            paper_bgcolor: "rgba(0,0,0,0)",
            font: { color: "white" },
            height: 250,
            margin: { t: 25, r: 25, l: 25, b: 25 }
        };

        Plotly.newPlot('gauge-chart', [gaugeData], layout);

        // Update confidence text
        document.getElementById('confidence-text').textContent = 
            `Confidence: ${(data.confidence * 100).toFixed(2)}%`;

        // Update indicators
        const indicatorsList = document.getElementById('indicators-list');
        indicatorsList.innerHTML = '';
        if (data.indicators.length > 0) {
            document.getElementById('indicators').style.display = 'block';
            data.indicators.forEach(indicator => {
                const div = document.createElement('div');
                div.className = 'indicator';
                div.textContent = indicator;
                indicatorsList.appendChild(div);
            });
        } else {
            document.getElementById('indicators').style.display = 'none';
        }

        // Update processed text
        document.getElementById('processed-text').textContent = data.processed_text;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('loading').style.display = 'none';
        alert('An error occurred while processing your request');
    });
}

function toggleProcessedText() {
    const processedText = document.getElementById('processed-text');
    processedText.style.display = processedText.style.display === 'none' ? 'block' : 'none';
} 