import React, { useState } from "react";

// Base URL pointing to backend API prefix
const API_URL = "http://localhost:5000/api";

function Register({ onRegisterSuccess }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");

  const handleRegister = async (e) => {
    e.preventDefault();
    setMessage("");
    const res = await fetch(`${API_URL}/signup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password })
    });
    const data = await res.json();
    if (res.ok) {
      setMessage("Registered! Please login.");
      onRegisterSuccess();
    } else {
      setMessage(data.error || "Registration failed");
    }
  };

  return (
    <form onSubmit={handleRegister} className="space-y-4">
      <h2 className="text-2xl font-bold">Register</h2>
      <input type="email" placeholder="Email" value={email} onChange={e => setEmail(e.target.value)} className="w-full px-3 py-2 border rounded" required />
      <input type="password" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} className="w-full px-3 py-2 border rounded" required />
      <button className="w-full bg-blue-500 text-white py-2 rounded">Register</button>
      <div className="text-sm text-red-500">{message}</div>
    </form>
  );
}

export default Register;
