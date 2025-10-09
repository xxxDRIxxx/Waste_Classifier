const confidenceSlider = document.getElementById("confidence");
const confValue = document.getElementById("conf_value");
const sourceSelect = document.getElementById("source");
const runBtn = document.getElementById("run_btn");
const uploadBtn = document.getElementById("upload_btn");
const fileInput = document.getElementById("file_input");
const outputImg = document.getElementById("output_img");

confidenceSlider.addEventListener("input", () => {
    confValue.innerText = confidenceSlider.value;
});

// Show file input only if Upload Image is selected
sourceSelect.addEventListener("change", () => {
    if(sourceSelect.value === "upload") {
        fileInput.style.display = "inline-block";
    } else {
        fileInput.style.display = "none";
    }
});

uploadBtn.addEventListener("click", () => fileInput.click());

// JS triggers Streamlit callback via Streamlit events
runBtn.addEventListener("click", () => {
    const source = sourceSelect.value;
    const confidence = parseFloat(confidenceSlider.value);

    // Send data to Streamlit
    window.parent.postMessage({
        type: "RUN_DETECTION",
        source: source,
        confidence: confidence,
        file: fileInput.files[0] || null
    }, "*");
});
