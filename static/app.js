// Agentic AI Academic Researcher 2.0 - Frontend Controller

let currentJobId = null;
let pollInterval = null;
let timerInterval = null;
let startTime = null;
let currentReportText = "";
let currentReportTopic = "";

// Dynamic Backend URL resolution:
// If Frontend is deployed as a standalone Render Static Site, it connects to the deployed Backend Web Service URL.
let BACKEND_URL = localStorage.getItem("researcher_backend_url") || "";

function getApiUrl(endpoint) {
    if (!BACKEND_URL) {
        return endpoint;
    }
    const cleanBase = BACKEND_URL.replace(/\/+$/, "");
    const cleanEndpoint = endpoint.startsWith("/") ? endpoint : `/${endpoint}`;
    return `${cleanBase}${cleanEndpoint}`;
}

// DOM Elements
const systemStatusEl = document.getElementById("systemStatus");
const dossierCountEl = document.getElementById("dossierCount");
const researchForm = document.getElementById("researchForm");
const queryInput = document.getElementById("queryInput");
const submitBtn = document.getElementById("submitBtn");
const progressSection = document.getElementById("progressSection");
const progressStatusText = document.getElementById("progressStatusText");
const timerDisplay = document.getElementById("timerDisplay");
const resultSection = document.getElementById("resultSection");
const resultTopic = document.getElementById("resultTopic");
const resultMeta = document.getElementById("resultMeta");
const tabRendered = document.getElementById("tabRendered");
const rawMarkdownCode = document.getElementById("rawMarkdownCode");
const downloadBtn = document.getElementById("downloadBtn");
const copyBtn = document.getElementById("copyBtn");
const toggleDrawerBtn = document.getElementById("toggleDrawerBtn");
const closeDrawerBtn = document.getElementById("closeDrawerBtn");
const reportsDrawer = document.getElementById("reportsDrawer");
const drawerOverlay = document.getElementById("drawerOverlay");
const reportsList = document.getElementById("reportsList");
const openConfigBtn = document.getElementById("openConfigBtn");
const closeConfigBtn = document.getElementById("closeConfigBtn");
const configModal = document.getElementById("configModal");
const saveConfigBtn = document.getElementById("saveConfigBtn");
const backendUrlInput = document.getElementById("backendUrlInput");
const groqKeyInput = document.getElementById("groqKeyInput");
const configFeedback = document.getElementById("configFeedback");

// Step elements
const steps = [
    document.getElementById("step1"),
    document.getElementById("step2"),
    document.getElementById("step3"),
    document.getElementById("step4")
];

// Phase descriptions
const phaseMessages = [
    "Phase 1: Literature Scout querying arXiv & Semantic Scholar...",
    "Phase 2: Ingesting open-access PDFs & layout-aware PyMuPDF chunking...",
    "Phase 3: Hybrid Retrieval (ChromaDB + BM25) & Reciprocal Rank Fusion...",
    "Phase 4: Research Strategist synthesizing comparative review & future directions..."
];

// Initialize on Load
document.addEventListener("DOMContentLoaded", () => {
    checkSystemHealth();
    loadReportsList();
    setupEventListeners();
});

function setupEventListeners() {
    // Research Form Submit
    researchForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;
        startResearch(query);
    });

    // Suggested Topic Chips
    document.querySelectorAll(".chip").forEach(chip => {
        chip.addEventListener("click", () => {
            queryInput.value = chip.getAttribute("data-query");
            queryInput.focus();
            showToast("Topic populated into input field", "info");
        });
    });

    // Tabs Navigation
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            const targetId = btn.getAttribute("data-tab");
            document.getElementById(targetId).classList.add("active");
        });
    });

    // Download Button (Client-side Blob - zero reload, instant download)
    downloadBtn.addEventListener("click", () => {
        if (!currentReportText) return;
        const safeName = (currentReportTopic || "research")
            .toLowerCase()
            .replace(/[^a-z0-9_-]/g, "_")
            .substring(0, 30);
        const filename = `research_${safeName}_${Date.now()}.md`;
        
        const blob = new Blob([currentReportText], { type: "text/markdown;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);

        showToast("Dossier downloaded successfully!", "success");
    });

    // Copy to Clipboard
    copyBtn.addEventListener("click", async () => {
        if (!currentReportText) return;
        try {
            await navigator.clipboard.writeText(currentReportText);
            const originalText = copyBtn.innerHTML;
            copyBtn.innerHTML = "✓ Copied!";
            showToast("Markdown copied to clipboard!", "success");
            setTimeout(() => {
                copyBtn.innerHTML = originalText;
            }, 2500);
        } catch (err) {
            showToast("Failed to copy to clipboard", "error");
        }
    });

    // Drawer Controls
    toggleDrawerBtn.addEventListener("click", openDrawer);
    closeDrawerBtn.addEventListener("click", closeDrawer);
    drawerOverlay.addEventListener("click", closeDrawer);

    // Config Modal Controls
    openConfigBtn.addEventListener("click", () => {
        if (backendUrlInput) {
            backendUrlInput.value = BACKEND_URL;
        }
        configModal.classList.remove("hidden");
        configFeedback.textContent = "";
    });
    closeConfigBtn.addEventListener("click", () => configModal.classList.add("hidden"));
    configModal.addEventListener("click", (e) => {
        if (e.target === configModal) configModal.classList.add("hidden");
    });

    // Save Config
    saveConfigBtn.addEventListener("click", saveSettings);
}

// System Health Polling
async function checkSystemHealth() {
    try {
        const res = await fetch(getApiUrl("/health"));
        if (!res.ok) throw new Error("Health check failed");
        const data = await res.json();

        dossierCountEl.textContent = data.reports_available || 0;

        if (data.groq_configured) {
            systemStatusEl.className = "status-badge status-online";
            systemStatusEl.innerHTML = `<span class="dot"></span><span class="status-text">🟢 Online (Groq Qwen 27B)</span>`;
        } else {
            systemStatusEl.className = "status-badge status-warning";
            systemStatusEl.innerHTML = `<span class="dot"></span><span class="status-text">⚠️ Missing Groq Key</span>`;
        }
    } catch (err) {
        systemStatusEl.className = "status-badge status-offline";
        systemStatusEl.innerHTML = `<span class="dot"></span><span class="status-text">🔴 Backend Disconnected</span>`;
        
        // If loaded on a static domain without a configured backend URL, nudge the user
        if (!BACKEND_URL && window.location.hostname.includes("onrender.com")) {
            showToast("⚠️ Configure your Render Backend URL in Settings", "warning");
        }
    }
}

// Start Research Job
async function startResearch(query) {
    // Reset state & UI
    currentReportTopic = query;
    submitBtn.disabled = true;
    submitBtn.textContent = "⏳ Investigating...";
    resultSection.classList.add("hidden");
    progressSection.classList.remove("hidden");

    // Stepper reset
    steps.forEach((s, idx) => {
        s.className = idx === 0 ? "step-item active" : "step-item";
    });
    progressStatusText.textContent = phaseMessages[0];

    // Start Timer
    startTime = Date.now();
    timerDisplay.textContent = "0.0s";
    clearInterval(timerInterval);
    timerInterval = setInterval(() => {
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        timerDisplay.textContent = `${elapsed}s`;
    }, 100);

    try {
        const res = await fetch(getApiUrl("/api/research"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: query })
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || "Failed to start research task. Check backend connection.");
        }

        const data = await res.json();
        currentJobId = data.job_id;

        // Start polling job status
        pollJobStatus(currentJobId);

    } catch (err) {
        handleResearchError(err.message);
    }
}

// Poll Job Status
function pollJobStatus(jobId) {
    let pollCount = 0;
    clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
        pollCount++;
        // Update visual stepper based on elapsed stages
        const stageIndex = Math.min(Math.floor(pollCount / 3), 3);
        steps.forEach((s, idx) => {
            if (idx < stageIndex) s.className = "step-item completed";
            else if (idx === stageIndex) s.className = "step-item active";
            else s.className = "step-item";
        });
        progressStatusText.textContent = phaseMessages[stageIndex];

        try {
            const res = await fetch(getApiUrl(`/api/research/${jobId}`));
            if (!res.ok) throw new Error("Status check failed");
            const job = await res.json();

            if (job.status === "completed") {
                clearInterval(pollInterval);
                clearInterval(timerInterval);
                handleResearchSuccess(job.result);
            } else if (job.status === "failed") {
                clearInterval(pollInterval);
                clearInterval(timerInterval);
                handleResearchError(job.error || "Autonomous research execution failed.");
            }
        } catch (err) {
            // Keep polling unless persistent failure
            if (pollCount > 120) { // 4 minutes timeout
                clearInterval(pollInterval);
                clearInterval(timerInterval);
                handleResearchError("Job polling timed out.");
            }
        }
    }, 2000);
}

// Success Handler
function handleResearchSuccess(resultText) {
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    submitBtn.disabled = false;
    submitBtn.textContent = "🚀 Launch Autonomous Research Crew";
    progressSection.classList.add("hidden");

    // All steps completed
    steps.forEach(s => s.className = "step-item completed");

    currentReportText = resultText;

    // Populate Results
    resultTopic.textContent = currentReportTopic;
    resultMeta.textContent = `Completed in ${elapsed}s • Verified Citations Grounded`;

    // Render Markdown via marked.js
    if (window.marked) {
        tabRendered.innerHTML = marked.parse(resultText);
    } else {
        tabRendered.textContent = resultText;
    }

    rawMarkdownCode.textContent = resultText;

    resultSection.classList.remove("hidden");
    resultSection.scrollIntoView({ behavior: "smooth", block: "start" });

    showToast("Research synthesis completed successfully!", "success");
    checkSystemHealth();
    loadReportsList();
}

// Error Handler
function handleResearchError(errMsg) {
    submitBtn.disabled = false;
    submitBtn.textContent = "🚀 Launch Autonomous Research Crew";
    progressSection.classList.add("hidden");
    clearInterval(timerInterval);
    clearInterval(pollInterval);

    showToast(`Error: ${errMsg}`, "error");
    alert(`Research Pipeline Error:\n\n${errMsg}`);
}

// Stored Reports Drawer
async function loadReportsList() {
    try {
        const res = await fetch(getApiUrl("/api/reports"));
        if (!res.ok) return;
        const reports = await res.json();

        dossierCountEl.textContent = reports.length;

        if (reports.length === 0) {
            reportsList.innerHTML = `<p class="empty-state">No dossiers saved yet. Run your first inquiry!</p>`;
            return;
        }

        reportsList.innerHTML = reports.map(r => `
            <div class="report-item" onclick="loadReport('${encodeURIComponent(r.filename)}')">
                <div class="report-name">📄 ${r.filename}</div>
                <div class="report-meta">
                    <span>${r.size_kb} KB</span>
                    <span>${new Date(r.modified).toLocaleDateString()}</span>
                </div>
            </div>
        `).join("");

    } catch (err) {
        reportsList.innerHTML = `<p class="empty-state">No connection to stored dossiers.</p>`;
    }
}

async function loadReport(filename) {
    try {
        const res = await fetch(getApiUrl(`/api/reports/${filename}`));
        if (!res.ok) throw new Error("Could not retrieve report");
        const data = await res.json();

        currentReportText = data.content;
        currentReportTopic = decodeURIComponent(filename).replace(".md", "");

        resultTopic.textContent = currentReportTopic;
        resultMeta.textContent = `Loaded from Stored Dossiers: ${data.filename}`;

        if (window.marked) {
            tabRendered.innerHTML = marked.parse(data.content);
        } else {
            tabRendered.textContent = data.content;
        }

        rawMarkdownCode.textContent = data.content;

        resultSection.classList.remove("hidden");
        closeDrawer();
        resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
        showToast(`Loaded dossier: ${data.filename}`, "info");

    } catch (err) {
        showToast("Error loading stored dossier", "error");
    }
}

function openDrawer() {
    reportsDrawer.classList.remove("hidden");
    drawerOverlay.classList.remove("hidden");
    loadReportsList();
}

function closeDrawer() {
    reportsDrawer.classList.add("hidden");
    drawerOverlay.classList.add("hidden");
}

// Settings & API Configuration
async function saveSettings() {
    const backendUrl = backendUrlInput ? backendUrlInput.value.trim() : "";
    const key = groqKeyInput.value.trim();

    // 1. Update Backend URL in localStorage
    if (backendUrl) {
        localStorage.setItem("researcher_backend_url", backendUrl);
        BACKEND_URL = backendUrl;
    } else if (backendUrl === "") {
        localStorage.removeItem("researcher_backend_url");
        BACKEND_URL = "";
    }

    // 2. If API Key provided, push to backend
    if (key) {
        try {
            const res = await fetch(getApiUrl("/api/config"), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ groq_api_key: key })
            });

            if (!res.ok) throw new Error("Failed to save key on backend");
            groqKeyInput.value = "";
        } catch (err) {
            configFeedback.className = "feedback-msg error";
            configFeedback.textContent = `Error saving key: ${err.message}`;
            return;
        }
    }

    configFeedback.className = "feedback-msg success";
    configFeedback.textContent = "Settings saved successfully!";
    
    setTimeout(() => {
        configModal.classList.add("hidden");
        checkSystemHealth();
        loadReportsList();
        showToast("Settings updated & reconnected", "success");
    }, 800);
}

// Toast Notifications
function showToast(msg, type = "info") {
    const container = document.getElementById("toastContainer");
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.textContent = msg;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}
