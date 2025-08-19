import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { useAuth } from '../App'

export default function NavBar() {
  const { apiKey, user, signout } = useAuth()
  const location = useLocation()
  const here = (p) => location.pathname === p

  return (
    <div className="nav">
      <div className="links">
        <Link className="pill" to="/">Dashboard</Link>
        <Link className="pill" to="/profile">Profile</Link>
        {user?.is_admin && <Link className="pill" to="/admin/users">Admin</Link>}
      </div>
      <div className="links">
        {apiKey ? (
          <>
            <span className="badge">{user?.email}</span>
            <button className="secondary" onClick={signout}>Sign out</button>
          </>
        ) : (
          <>
            <Link to="/signin"><button className={here('/signin') ? '' : 'secondary'}>Sign in</button></Link>
            <Link to="/signup"><button>Sign up</button></Link>
          </>
        )}
      </div>
    </div>
  )
}
