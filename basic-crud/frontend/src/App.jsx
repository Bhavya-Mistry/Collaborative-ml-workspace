import React, { useEffect, useState, createContext, useContext } from 'react'
import { Routes, Route, Navigate, useNavigate } from 'react-router-dom'
import { getMe, signin as apiSignin, signup as apiSignup } from './api'
import Signin from "./pages/Signin.jsx";
import Signup from "./pages/Signup.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Profile from "./pages/Profile.jsx";
import AdminUsers from "./pages/AdminUsers.jsx";
import NavBar from "./components/NavBar.jsx";


export const AuthContext = createContext(null)
export function useAuth(){ return useContext(AuthContext) }

function Protected({ children }) {
  const { apiKey, user } = useAuth()
  if (!apiKey) return <Navigate to="/signin" replace />
  if (apiKey && user === null) return <div className="container"><div className="card">Loading...</div></div>
  return children
}

function AdminOnly({ children }) {
  const { user } = useAuth()
  if (!user?.is_admin) return <Navigate to="/" replace />
  return children
}

export default function App() {
  const [apiKey, setApiKey] = useState(localStorage.getItem('apiKey') || '')
  const [user, setUser] = useState(null)
  const navigate = useNavigate()

  async function refreshMe() {
    if (!apiKey) { setUser(null); return }
    try {
      const me = await getMe()
      setUser(me)
    } catch (e) {
      console.error(e)
      setUser(null)
      localStorage.removeItem('apiKey')
      setApiKey('')
    }
  }

  useEffect(() => { refreshMe() }, [apiKey])

  async function handleSignin(email, password) {
    const res = await apiSignin(email, password)
    localStorage.setItem('apiKey', res.api_key)
    setApiKey(res.api_key)
    navigate('/')
  }
  async function handleSignup(email, password) {
    const res = await apiSignup(email, password)
    localStorage.setItem('apiKey', res.api_key)
    setApiKey(res.api_key)
    navigate('/')
  }
  function signout() {
    localStorage.removeItem('apiKey'); setApiKey(''); setUser(null); navigate('/signin')
  }

  const ctx = { apiKey, user, setUser, signout, handleSignin, handleSignup, refreshMe }

  return (
    <AuthContext.Provider value={ctx}>
      <div className="container">
        <NavBar />
        <Routes>
          <Route path="/signin" element={<Signin />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/" element={<Protected><Dashboard /></Protected>} />
          <Route path="/profile" element={<Protected><Profile /></Protected>} />
          <Route path="/admin/users" element={<Protected><AdminOnly><AdminUsers /></AdminOnly></Protected>} />
          <Route path="*" element={<Navigate to="/" replace/>} />
        </Routes>
      </div>
    </AuthContext.Provider>
  )
}
