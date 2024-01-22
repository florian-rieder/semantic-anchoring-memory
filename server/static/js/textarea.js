// see https://codepen.io/awolkers/pen/ZEOmKxO
function setTextAreaHeight(elem) {
    const style = getComputedStyle(elem, null);
    const verticalBorders = Math.round(parseFloat(style.borderTopWidth) + parseFloat(style.borderBottomWidth));
    const maxHeight = parseFloat(style.maxHeight) || 300;

    elem.style.height = "auto";

    const newHeight = elem.scrollHeight + verticalBorders;

    elem.style.overflowY = newHeight > maxHeight ? "auto" : "hidden";
    elem.style.height = Math.min(newHeight, maxHeight) + "px";
}

const textarea = document.querySelector("textarea.textarea-expand")

textarea.addEventListener("input", (e) => {
    let sendButton = document.getElementById("send");
    // Only enable the send button if there is something in the textarea
    sendButton.disabled = e.target.value.trim() == "";

    setTextAreaHeight(e.target);
    e.preventDefault();
});

// Submit on enter, Shift+Enter inserts new line
textarea.addEventListener("keydown", submitOnEnter);
// see https://stackoverflow.com/a/49389811/10914628
function submitOnEnter(event) {
    if (event.which === 13 && !event.shiftKey) {
        if (!event.repeat) {
            const newEvent = new Event("submit", {cancelable: true});
            event.target.form.dispatchEvent(newEvent);
        }

        event.preventDefault(); // Prevents the addition of a new line in the text field
    }
}


setTextAreaHeight(textarea);
// Set the focus on the textarea, so that you can immediately start typing
textarea.focus();
