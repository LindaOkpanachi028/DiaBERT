document.getElementById("classify").addEventListener("click", async () => {
    const text = document.getElementById("inputText").value.trim();
    const classifyBtn = document.getElementById("classify");

    // Change button color to red and text to "Classifying..."
    classifyBtn.style.backgroundColor = "#f44336";  // Red when clicked
    classifyBtn.textContent = "Classifying...";     // Show progress

    if (!text) {
        alert("Please enter some text.");
        classifyBtn.textContent = "Classify Text";  // Reset the text after alert
        return;
    }

    const serverUrl = "https://diabert-1-3vgv.onrender.com/predict";

    try {
        const response = await fetch(serverUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });

        if (!response.ok) {
            throw new Error(`Server responded with status ${response.status}`);
        }

        const result = await response.json();

        // Handle unrelated query
        if (result.is_relevant === false) {
            document.getElementById("result").innerHTML =
                `<strong style="font-size: 18px;">Unrelated Query:</strong> ${result.message || "The input is unrelated to diabetes."}`;
            document.getElementById("confidenceChart").style.display = "none";
            document.getElementById("explanationSection").style.display = "none";
            classifyBtn.textContent = "Classify Text";  // Reset after unrelated
            return;
        }

        // Handling prediction result
        const predLabel = result.predicted_label.toLowerCase();
        let color = predLabel === "false" ? "red" : predLabel === "real" ? "green" : "#ff9800";  // orange for partially_true
        let labelText = predLabel.replace("_", " ").toUpperCase();

        document.getElementById("result").innerHTML =
            `<strong style="font-size: 20px;">PREDICTION:</strong> 
             <span style="color: ${color}; font-weight: bold; font-size: 22px;">${labelText}</span>`;

        renderChart(result.probabilities);

        if (result.explanation && result.explanation !== "The model's attribution scores were too low for a reliable explanation.") {
            displayExplainability(text, result.key_words, result.explanation);
        } else {
            document.getElementById("explanationSection").style.display = "block";
            document.getElementById("highlightedText").innerHTML =
                "<strong>Explanation Not Available:</strong> The model did not rely on specific words strongly enough.";
            document.getElementById("explanationText").innerHTML = "";
        }

        // Reset the button text after prediction
        classifyBtn.textContent = "Classify Text";

    } catch (error) {
        console.error("Error:", error);
        document.getElementById("result").innerHTML =
            `<strong style="color: red;">Server Error:</strong> ${error.message || "Please try again later."}`;
        classifyBtn.textContent = "Classify Text";  // Reset on error too
    }
});

function renderChart(probabilities) {
    const ctx = document.getElementById("confidenceChart").getContext("2d");

    if (window.confidenceChart && typeof window.confidenceChart.destroy === "function") {
        window.confidenceChart.destroy();
    }

    let values = [
        parseFloat(probabilities.real || 0),
        parseFloat(probabilities.false || 0),
        parseFloat(probabilities.partially_true || 0)
    ];

    const minVisibleValue = 1;
    values = values.map(val => Math.max(val, minVisibleValue));

    window.confidenceChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["Real", "False", "Partially True"],
            datasets: [{
                label: "Confidence Score (%)",
                data: values,
                backgroundColor: ["#4caf50", "#f44336", "#ff9800"],  // Green, Red, Orange
                borderWidth: 1,
            }],
        },
        options: {
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });

    document.getElementById("confidenceChart").style.display = "block";
}


//Merge subtokens like "##meric" into "Turmeric"
function mergeSubtokens(tokens) {
    if (!Array.isArray(tokens)) return [];  // 🛡️ Safety check
    const merged = [];
    let current = "";

    for (let token of tokens) {
        if (token.startsWith("##")) {
            current += token.replace("##", "");
        } else {
            if (current) merged.push(current);
            current = token;
        }
    }
    if (current) merged.push(current);
    return merged;
}



function displayExplainability(text, keyWords, explanation) {
    const explanationSection = document.getElementById("explanationSection");
    const highlightedTextContainer = document.getElementById("highlightedText");
    const explanationText = document.getElementById("explanationText");

    if (!keyWords || keyWords.length === 0) {
        highlightedTextContainer.innerHTML =
            `<strong>Highlighted Text:</strong> ${text} (No specific keywords detected)`;
    } else {
        let highlightedText = text;
        const cleanKeywords = mergeSubtokens(keyWords);

        cleanKeywords.forEach(word => {
            const escapedWord = word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const regex = new RegExp(`\\b${escapedWord}\\b`, "gi");
            highlightedText = highlightedText.replace(regex, match => 
                `<span style="background-color: yellow; font-weight: bold;">${match}</span>`
            );
        });

        highlightedTextContainer.innerHTML = `<strong>Highlighted Text:</strong> ${highlightedText}`;
    }

    explanationText.innerHTML = `<strong>Explanation:</strong> ${explanation}`;
    explanationSection.style.display = "block";
}

// Delaying classify button to ensure backend loads resources
window.onload = () => {
    const classifyBtn = document.getElementById("classify");
    classifyBtn.disabled = true;
    classifyBtn.textContent = "Loading model...";

    setTimeout(() => {
        classifyBtn.disabled = false;
        classifyBtn.textContent = "Classify Text";
    }, 2000); // 2 second buffer for resource loading
};

document.getElementById("clear").addEventListener("click", () => {
    document.getElementById("inputText").value = "";
    document.getElementById("result").innerHTML = "";
    document.getElementById("confidenceChart").style.display = "none";
    document.getElementById("explanationSection").style.display = "none";

    // Destroy chart
    if (window.confidenceChart && typeof window.confidenceChart.destroy === "function") {
        window.confidenceChart.destroy();
    }

    // Reset classify button color and text
    const classifyBtn = document.getElementById("classify");
    classifyBtn.style.backgroundColor = "";
    classifyBtn.textContent = "Classify Text";
});
