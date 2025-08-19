import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

// Base URL pointing to backend API prefix
const API_URL = "http://localhost:5000/api";

function Login({ onLoginSuccess }) {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");

  const handleLogin = async (e) => {
    e.preventDefault();
    setMessage("");
    const res = await fetch(`${API_URL}/signin`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password })
    });
    const data = await res.json();
    if (res.ok) {
      onLoginSuccess({ email, api_key: data.api_key });
      navigate('/dashboard');
    } else {
      setMessage(data.error || "Login failed");
    }
  };

  return (
    <div>
      <form onSubmit={handleLogin} className="space-y-4">
        <h2 className="text-2xl font-bold">Login</h2>
        <input type="email" placeholder="Email" value={email} onChange={e => setEmail(e.target.value)} className="w-full px-3 py-2 border rounded" required />
        <input type="password" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} className="w-full px-3 py-2 border rounded" required />
        <button className="w-full bg-blue-500 text-white py-2 rounded">Login</button>
        <div className="text-sm text-red-500">{message}</div>
      </form>
      <div className="mt-2 text-sm">
        Don't have an account? <a href="/register" className="text-blue-600">Register</a>
      </div>
    </div>
  );
}

export default Login;
