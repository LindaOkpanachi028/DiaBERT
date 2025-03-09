// Extract all text from the webpage
function extractPageText() {
    return document.body.innerText; // You can refine this to target specific elements
}

// Send extracted text to the background script for classification
function classifyPageText() {
    const text = extractPageText();

    chrome.runtime.sendMessage(
        { action: "classifyText", text },
        (response) => {
            console.log("Classification Result:", response.result);
            alert(`Text classification result: ${response.result}`);
        }
    );
}

// Automatically classify the page when the script is loaded
classifyPageText();
