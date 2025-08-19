import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import Register from "./Register";
import Login from "./Login";
import Dashboard from "./DashboardFixed";
import "./index.css";

function App() {
  const [user, setUser] = useState(null);

  return (
    <Router>
      <div className="min-h-screen flex items-center justify-center bg-gray-100">
        <div className="bg-white p-8 rounded shadow w-full max-w-md">
          <Routes>
            <Route path="/register" element={<Register onRegisterSuccess={() => window.location.replace('/login')} />} />
            <Route path="/login" element={<Login onLoginSuccess={setUser} />} />
            <Route path="/dashboard" element={user ? <Dashboard user={user} onLogout={() => setUser(null)} /> : <Navigate to="/login" />} />
            <Route path="*" element={<Navigate to={user ? "/dashboard" : "/login"} />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
