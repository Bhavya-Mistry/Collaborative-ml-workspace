import React, { useState } from 'react'
import { useAuth } from '../App'
import { Link } from 'react-router-dom'

export default function Signup() {
  const { handleSignup } = useAuth()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(false)

  async function submit(e) {
    e.preventDefault()
    setErr(''); setLoading(true)
    try {
      await handleSignup(email, password)
    } catch (e) {
      setErr(e?.data?.error || e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card" style={{maxWidth: 480, margin: '8vh auto'}}>
      <h2 style={{marginTop:0}}>Sign up</h2>
      <form onSubmit={submit}>
        <div className="field">
          <label>Email</label>
          <input value={email} onChange={e=>setEmail(e.target.value)} placeholder="you@example.com" />
        </div>
        <div className="field">
          <label>Password</label>
          <input type="password" value={password} onChange={e=>setPassword(e.target.value)} placeholder="••••••••" />
        </div>
        {err && <div className="help" style={{color:'#fecaca'}}>{err}</div>}
        <div className="row" style={{justifyContent:'space-between', alignItems:'center'}}>
          <button disabled={loading}>{loading?'Creating...':'Create account'}</button>
          <Link to="/signin">Have an account? Sign in</Link>
        </div>
      </form>
    </div>
  )
}
