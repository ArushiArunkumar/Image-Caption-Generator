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
