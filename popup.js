document.getElementById("classify").addEventListener("click", async () => {
    const text = document.getElementById("inputText").value.trim();

    if (!text) {
        alert("Please enter some text.");
        return;
    }

    console.log("Entered Text:", text);

    const serverUrl = "http://127.0.0.1:5000/predict";

    try {
        const response = await fetch(serverUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });

        const result = await response.json();
        console.log("Server response:", result);

        if (!result || typeof result !== "object") {
            throw new Error("Invalid server response.");
        }

        // Handle unrelated query response
        if (result.is_relevant === false) {
            document.getElementById("result").innerHTML = 
                `<strong style="font-size: 18px;">Unrelated Query:</strong> ${result.message || "The input is unrelated to diabetes."}`;
            document.getElementById("confidenceChart").style.display = "none";
            document.getElementById("explanationSection").style.display = "none";
            return;
        }

        // ✅ Ensure probabilities exist before accessing them
        if (!result.probabilities || typeof result.probabilities.false === "undefined" || typeof result.probabilities.real === "undefined") {
            console.error("Invalid probabilities:", result.probabilities);
            document.getElementById("result").innerHTML = 
                `<strong style="color: red;">Error:</strong> Invalid response from the model.`;
            document.getElementById("confidenceChart").style.display = "none";
            return;
        }

        //
        let predictionText = result.predicted_label.toLowerCase() === "false" 
            ? '<span style="color: red; font-weight: bold; font-size: 22px;">FALSE</span>'
            : '<span style="color: green; font-weight: bold; font-size: 22px;">TRUE</span>';

        document.getElementById("result").innerHTML =
            `<strong style="font-size: 20px;">PREDICTION:</strong> ${predictionText}`;

        // Render confidence chart
        renderChart(result.probabilities);

        // Handle explainability
        if (result.explanation && result.explanation !== "The model's attribution scores were too low for a reliable explanation.") {
            displayExplainability(text, result.key_words, result.explanation);
        } else {
            document.getElementById("explanationSection").style.display = "block";
            document.getElementById("highlightedText").innerHTML = 
                "<strong>Explanation Not Available:</strong> The model did not rely on specific words strongly enough to generate an explanation.";
            document.getElementById("explanationText").innerHTML = "";
        }

    } catch (error) {
        console.error("Error:", error);
        document.getElementById("result").innerHTML =
            `<strong style="color: red;">Server Error:</strong> Please try again later.`;
    }
});

function renderChart(probabilities) {
    const ctx = document.getElementById("confidenceChart").getContext("2d");

    if (window.confidenceChart && typeof window.confidenceChart.destroy === "function") {
        window.confidenceChart.destroy();
    }
    window.confidenceChart = null;

    console.log("Rendering chart with probabilities:", probabilities);

    if (!probabilities || typeof probabilities.real === "undefined" || typeof probabilities.false === "undefined") {
        console.error("Invalid probabilities:", probabilities);
        document.getElementById("confidenceChart").style.display = "none";
        alert("Error: Unable to display chart.");
        return;
    }

    let realConfidence = probabilities.real;
    let falseConfidence = probabilities.false;

    if (realConfidence <= 1 && falseConfidence <= 1) {
        realConfidence *= 100;
        falseConfidence *= 100;
    }

    const minVisibleValue = 1;
    realConfidence = Math.max(realConfidence, minVisibleValue);
    falseConfidence = Math.max(falseConfidence, minVisibleValue);

    //Create a new chart instance
    window.confidenceChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["Real", "False"],
            datasets: [{
                label: "Confident Score(%)",
                data: [realConfidence, falseConfidence],
                backgroundColor: ["#4caf50", "#f44336"], // Green for True, Red for False
                borderWidth: 1,
            }],
        },
        options: {
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                },
            },
        },
    });

    document.getElementById("confidenceChart").style.display = "block";
}


//Function to Highlight Keywords and Display Explanation
function displayExplainability(text, keyWords, explanation) {
    const explanationSection = document.getElementById("explanationSection");
    const highlightedTextContainer = document.getElementById("highlightedText");
    const explanationText = document.getElementById("explanationText");

    if (!keyWords || keyWords.length === 0) {
        highlightedTextContainer.innerHTML = 
            `<strong>Highlighted Text:</strong> ${text} (No specific keywords detected)`;
    } else {
        let highlightedText = text;
        keyWords.forEach(word => {
            word = word.replace(/\./g, "");
            const regex = new RegExp(`\\b${word}\\b`, "gi");
            highlightedText = highlightedText.replace(regex, `<mark>${word}</mark>`);
        });

        highlightedTextContainer.innerHTML = `<strong>Highlighted Text:</strong> ${highlightedText}`;
    }

    explanationText.innerHTML = `<strong>Explanation:</strong> ${explanation}`;

    explanationSection.style.display = "block";
}
