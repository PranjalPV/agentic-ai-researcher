// ====================================================================
// Agentic AI Academic Researcher - Production API Configuration
// ====================================================================
// When deploying your frontend separately on Render (Static Site),
// paste your Render Backend Web Service URL below:
const PRODUCTION_BACKEND_URL = "https://agentic-ai-researcher-2w48.onrender.com";

window.APP_CONFIG = {
    // Returns the appropriate backend URL based on environment
    getBackendUrl: function () {
        // 1. Local development
        if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
            return "http://localhost:8000";
        }
        
        // 2. Custom override from localStorage (if configured)
        const customUrl = localStorage.getItem("researcher_backend_url");
        if (customUrl) {
            return customUrl.replace(/\/+$/, "");
        }

        // 3. If hosted on the same domain as backend, use relative paths
        if (window.location.origin.replace(/\/+$/, "") === PRODUCTION_BACKEND_URL.replace(/\/+$/, "")) {
            return "";
        }

        // 4. Default production backend for standalone static frontend
        return PRODUCTION_BACKEND_URL.replace(/\/+$/, "");
    }
};
