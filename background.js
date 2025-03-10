async function classifyTextServer(text) {
    const serverUrl = "https://diabert-jxbe.onrender.com/predict";

    try {
        const response = await fetch(serverUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });

        if (!response.ok) {
            throw new Error(`Server responded with status: ${response.status}`);
        }

        const result = await response.json();

        console.log(`Prediction: ${result.predicted_label} (Probabilities:`, result.probabilities, ")");

        return {
            prediction: result.predicted_label, // 
            confidence: result.probabilities[result.predicted_label], // 
            probabilities: result.probabilities,
            explanation: result.explanation || "No explanation provided.",
        };
    } catch (error) {
        console.error("Error communicating with the server:", error.message);
        return null;
    }
}

chrome.runtime.onMessage.addListener(async (request, sender, sendResponse) => {
    if (request.action === "classifyText") {
        const result = await classifyTextServer(request.text);
        sendResponse({ result });
    }
    // Important for asynchronous message passing in Chrome
    return true;
});
