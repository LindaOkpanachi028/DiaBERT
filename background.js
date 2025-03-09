async function classifyTextServer(text) {
    const serverUrl = "http://127.0.0.1:5000/predict";

    try {
        const response = await fetch(serverUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });

        const result = await response.json();

        console.log(`Prediction: ${result.prediction} (Confidence: ${result.confidence})`);
        return {
            prediction: result.prediction,
            confidence: result.confidence,
            probabilities: result.probabilities, // Add probabilities for the chart
        };
    } catch (error) {
        console.error("Error communicating with the server:", error);
        return null;
    }
}

chrome.runtime.onMessage.addListener(async (request, sender, sendResponse) => {
    if (request.action === "classifyText") {
        const result = await classifyTextServer(request.text);
        sendResponse({ result });
    }
});



