// static/script.js
async function classifyText() {
    const input = document.getElementById("inputText").value;
    const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input })
    });
    const data = await res.json();
    document.getElementById("output").innerText =
        `Categoría: ${data.category} (en 0.${data.elapsed_ms} ms)`;
}

async function fetchReport() {
    const res = await fetch("/report");
    const data = await res.json();

    const tbody = document.querySelector("#reportTable tbody");
    tbody.innerHTML = "";

    for (let label in data) {
        if (["accuracy", "macro avg", "weighted avg"].includes(label) || !data[label].precision) continue;
        const row = document.createElement("tr");

        row.innerHTML = `
            <td>${label}</td>
            <td>${(data[label].precision * 100).toFixed(2)}%</td>
            <td>${(data[label].recall * 100).toFixed(2)}%</td>
            <td>${(data[label]['f1-score'] * 100).toFixed(2)}%</td>
            <td>${data[label].support}</td>
        `;
        tbody.appendChild(row);
    }
}

window.onload = fetchReport;