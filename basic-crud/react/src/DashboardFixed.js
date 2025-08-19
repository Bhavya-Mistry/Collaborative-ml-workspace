import React from "react";

function Dashboard({ user, onLogout }) {
  if (!user) return null;

  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Dashboard</h2>
      <div className="mb-2">Welcome, <span className="font-semibold">{user.email}</span></div>
      <div className="mb-2">Credits: <span className="font-semibold">{user.credits ?? 0}</span></div>
      <button onClick={onLogout} className="bg-red-500 text-white py-2 px-4 rounded mb-4">Logout</button>
    </div>
  );
}

export default Dashboard;
