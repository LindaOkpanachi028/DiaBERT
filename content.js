// Extract all text from the webpage
function extractPageText() {
    return document.body.innerText.trim(); // Trim to avoid empty spaces
}

// Send extracted text to the background script for classification
function classifyPageText() {
    const text = extractPageText();

    if (!text) {
        alert("No meaningful text found on this page to classify.");
        return;
    }

    chrome.runtime.sendMessage(
        { action: "classifyText", text },
        (response) => {
            if (!response || !response.result) {
                alert("Error: Unable to classify the text. Please try again.");
                return;
            }

            const { prediction, confidence, explanation } = response.result;

            alert(`
                **Classification Result**:
                - Prediction: ${prediction}
                - Confidence: ${confidence.toFixed(2)}%
                - Explanation: ${explanation}
            `);
        }
    );
}

// Automatically classify the page when the script is loaded
classifyPageText();
