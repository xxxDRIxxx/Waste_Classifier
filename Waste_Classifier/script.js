const confidenceSlider = document.getElementById("confidence");
const confValue = document.getElementById("conf_value");
const sourceSelect = document.getElementById("source");
const actionBtn = document.getElementById("action_btn");

confidenceSlider.addEventListener("input", () => {
    confValue.innerText = confidenceSlider.value;
});

actionBtn.addEventListener("click", () => {
    const source = sourceSelect.value;
    const confidence = confidenceSlider.value;
    alert(`Running detection on ${source} with confidence ${confidence}`);
    // Here we can trigger Streamlit callbacks using custom events if needed
});
