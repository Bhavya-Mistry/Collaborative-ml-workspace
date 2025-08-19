// Simple vanilla-JS front-end that talks to Flask backend under /api/*
// It supports signup, signin, showing current user info, and logout.
// Auth persistence via localStorage("api_key").

const API_BASE = "http://localhost:5000/api";

// Sections
const registerSection = document.getElementById("register-section");
const loginSection    = document.getElementById("login-section");
const dashboardSection= document.getElementById("dashboard-section");

// Forms & elements
const registerForm = document.getElementById("register-form");
const loginForm    = document.getElementById("login-form");
const modelForm    = document.getElementById("model-form"); // may be hidden if backend lacks

// Links / buttons
const showLoginLink    = document.getElementById("show-login");
const showRegisterLink = document.getElementById("show-register");
const logoutBtn        = document.getElementById("logout-btn");

// Message areas
const registerMsg  = document.getElementById("register-message");
const loginMsg     = document.getElementById("login-message");
const dashboardUser= document.getElementById("dashboard-user");

function switchView(view) {
  // hide all then show requested
  [registerSection, loginSection, dashboardSection].forEach(s => s.classList.add("hidden"));
  if (view === "register") registerSection.classList.remove("hidden");
  else if (view === "login") loginSection.classList.remove("hidden");
  else if (view === "dashboard") dashboardSection.classList.remove("hidden");
}

// -------- Register --------
registerForm.addEventListener("submit", async e => {
  e.preventDefault();
  registerMsg.textContent = "";
  const email = document.getElementById("reg-email").value.trim();
  const password = document.getElementById("reg-password").value;
  try {
    const res = await fetch(`${API_BASE}/signup`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ email, password })
    });
    const data = await res.json();
    if (res.ok) {
      registerMsg.className = "text-green-600 text-sm mt-2";
      registerMsg.textContent = "Registered successfully. Please login.";
      switchView("login");
    } else {
      registerMsg.className = "text-red-600 text-sm mt-2";
      registerMsg.textContent = data.error || "Registration failed";
    }
  } catch(err) {
    registerMsg.className = "text-red-600 text-sm mt-2";
    registerMsg.textContent = "Network error";
  }
});

// -------- Login --------
loginForm.addEventListener("submit", async e => {
  e.preventDefault();
  loginMsg.textContent = "";
  const email = document.getElementById("login-email").value.trim();
  const password = document.getElementById("login-password").value;
  try {
    const res = await fetch(`${API_BASE}/signin`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ email, password })
    });
    const data = await res.json();
    if (res.ok) {
      localStorage.setItem("api_key", data.api_key);
      await loadDashboard();
    } else {
      loginMsg.className = "text-red-600 text-sm mt-2";
      loginMsg.textContent = data.error || "Login failed";
    }
  } catch(err) {
    loginMsg.className = "text-red-600 text-sm mt-2";
    loginMsg.textContent = "Network error";
  }
});

// ----- Dashboard helpers -----
async function loadDashboard() {
  const apiKey = localStorage.getItem("api_key");
  if (!apiKey) { switchView("login"); return; }
  try {
    const res = await fetch(`${API_BASE}/me`, { headers: {"X-API-Key": apiKey }});
    if (!res.ok) { throw new Error("auth"); }
    const user = await res.json();
    dashboardUser.innerHTML = `Logged in as <span class="font-semibold">${user.email}</span> | Credits: <span class="font-semibold">${user.credits}</span>`;
    switchView("dashboard");
  } catch(err) {
    localStorage.removeItem("api_key");
    switchView("login");
  }
}

logoutBtn.addEventListener("click", () => {
  localStorage.removeItem("api_key");
  switchView("login");
});

// Model CRUD – optional, only if backend provides /models endpoints
if (modelForm) {
  modelForm.addEventListener("submit", async e => {
    e.preventDefault();
    const apiKey = localStorage.getItem("api_key");
    if (!apiKey) return;
    const name = document.getElementById("model-name").value.trim();
    const version = document.getElementById("model-version").value.trim();
    const res = await fetch(`${API_BASE}/models`, {
      method: "POST",
      headers: {"Content-Type": "application/json", "X-API-Key": apiKey},
      body: JSON.stringify({ name, version })
    });
    if (res.ok) {
      document.getElementById("model-name").value = "";
      document.getElementById("model-version").value = "";
      fetchModels();
    }
  });
}

async function fetchModels() {
  const apiKey = localStorage.getItem("api_key");
  if (!apiKey) return;
  const list = document.getElementById("models-list");
  list.innerHTML = "";
  const res = await fetch(`${API_BASE}/models`, { headers: {"X-API-Key": apiKey}});
  if (!res.ok) return;
  const models = await res.json();
  models.forEach(m => {
    const li = document.createElement("li");
    li.className = "flex justify-between items-center p-2 border rounded";
    li.innerHTML = `<span>${m.name} (${m.version})</span><button class="text-red-600" data-id="${m.id}">Delete</button>`;
    list.appendChild(li);
  });
  list.addEventListener("click", async e => {
    const btn = e.target.closest("button[data-id]");
    if (!btn) return;
    const id = btn.dataset.id;
    await fetch(`${API_BASE}/models/${id}`, { method: "DELETE", headers: {"X-API-Key": localStorage.getItem("api_key")}});
    fetchModels();
  }, { once:true });
}

// -------- View switching links --------
showRegisterLink.addEventListener("click", e => { e.preventDefault(); switchView("register"); });
showLoginLink.addEventListener("click", e => { e.preventDefault(); switchView("login"); });

// -------- Init on load --------
window.addEventListener("DOMContentLoaded", () => {
  const apiKey = localStorage.getItem("api_key");
  if (apiKey) loadDashboard();
  else switchView("login");
});
