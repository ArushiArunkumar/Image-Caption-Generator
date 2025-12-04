// Smooth scroll for internal navigation (optional)
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener("click", function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute("href")).scrollIntoView({
            behavior: "smooth"
        });
    });
});

// Image click zoom (basic lightbox effect)
document.querySelectorAll("img.chart, .sample-block img").forEach(img => {
    img.addEventListener("click", () => {
        const overlay = document.createElement("div");
        overlay.style.position = "fixed";
        overlay.style.top = "0";
        overlay.style.left = "0";
        overlay.style.width = "100%";
        overlay.style.height = "100%";
        overlay.style.background = "rgba(0,0,0,0.85)";
        overlay.style.display = "flex";
        overlay.style.alignItems = "center";
        overlay.style.justifyContent = "center";
        overlay.style.cursor = "zoom-out";
        overlay.style.zIndex = "5000";

        const zoomImg = document.createElement("img");
        zoomImg.src = img.src;
        zoomImg.style.maxWidth = "90%";
        zoomImg.style.maxHeight = "90%";
        zoomImg.style.borderRadius = "10px";

        overlay.appendChild(zoomImg);
        document.body.appendChild(overlay);

        overlay.addEventListener("click", () => overlay.remove());
    });
});

// Small fade-in animation for content
window.addEventListener("DOMContentLoaded", () => {
    const content = document.querySelector(".content");
    if (content) {
        content.style.opacity = 0;
        content.style.transition = "opacity 0.6s ease";
        setTimeout(() => (content.style.opacity = 1), 50);
    }
});
// =======================
// Image preview
// =======================
document.getElementById("imageInput")?.addEventListener("change", function () {
    const file = this.files[0];
    const preview = document.getElementById("preview");

    if (file) {
        preview.src = URL.createObjectURL(file);
        preview.style.display = "block";
    }
});

// =======================
// Generate caption via Render API
// =======================
async function generateCaption() {
    const fileInput = document.getElementById("imageInput");
    const captionBox = document.getElementById("captionBox");

    if (!fileInput.files[0]) {
        captionBox.innerText = "Please upload an image first.";
        return;
    }

    captionBox.innerText = "Generating caption...";

    const formData = new FormData();
    formData.append("image", fileInput.files[0]);

    // Replace with your Render API endpoint:
    const API_URL = "https://RENDER-APP.onrender.com/caption";

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        captionBox.innerText = data.caption
            ? data.caption
            : "Error: Unable to generate caption.";

    } catch (err) {
        captionBox.innerText = "API error. Please check your backend deployment.";
    }
}