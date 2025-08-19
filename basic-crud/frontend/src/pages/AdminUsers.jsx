import React, { useEffect, useState } from 'react'
import { adminListUsers, adminCreateUser, adminUpdateUser, adminDeleteUser } from '../api'
import { useAuth } from '../App'   // ← (1) import auth to use signout()

function UserForm({ initial = {}, onSave, saving, submitLabel='Create' }) {
  const [form, setForm] = useState({
    email: '', password: '', first_name:'', last_name:'', phone:'', address:'', city:'', country:'', credits:0, is_admin:false,
    ...initial
  })
  useEffect(()=>{ setForm(s => ({...s, ...initial})) }, [initial])
  function setField(k, v){ setForm(s => ({...s, [k]: v})) }
  function submit(e){ e.preventDefault(); onSave(form) }
  return (
    <form onSubmit={submit}>
      <div className="row">
        <div className="field"><label>Email</label><input value={form.email||''} onChange={e=>setField('email', e.target.value)} required /></div>
        <div className="field"><label>Password</label><input type="password" value={form.password||''} onChange={e=>setField('password', e.target.value)} placeholder="(leave blank to keep)" /></div>
        <div className="field"><label>First name</label><input value={form.first_name||''} onChange={e=>setField('first_name', e.target.value)} /></div>
        <div className="field"><label>Last name</label><input value={form.last_name||''} onChange={e=>setField('last_name', e.target.value)} /></div>
        <div className="field"><label>Phone</label><input value={form.phone||''} onChange={e=>setField('phone', e.target.value)} /></div>
        <div className="field" style={{flex:1}}><label>Address</label><input value={form.address||''} onChange={e=>setField('address', e.target.value)} /></div>
        <div className="field"><label>City</label><input value={form.city||''} onChange={e=>setField('city', e.target.value)} /></div>
        <div className="field"><label>Country</label><input value={form.country||''} onChange={e=>setField('country', e.target.value)} /></div>
        <div className="field"><label>Credits</label><input type="number" value={form.credits||0} onChange={e=>setField('credits', e.target.value)} /></div>
        <div className="field"><label>Admin?</label>
          <select value={form.is_admin? 'true':'false'} onChange={e=>setField('is_admin', e.target.value === 'true')}>
            <option value="false">No</option>
            <option value="true">Yes</option>
          </select>
        </div>
      </div>
      <div className="row" style={{justifyContent:'flex-end'}}>
        <button disabled={saving}>{saving? 'Saving...': submitLabel}</button>
      </div>
    </form>
  )
}

export default function AdminUsers() {
  const [items, setItems] = useState([])
  const [page, setPage] = useState(1)
  const [perPage, setPerPage] = useState(10)
  const [q, setQ] = useState('')
  const [total, setTotal] = useState(0)
  const [pages, setPages] = useState(1)
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(true)

  const [creating, setCreating] = useState(false)
  const [saving, setSaving] = useState(false)
  const [editing, setEditing] = useState(null)

  const { signout } = useAuth()  // ← (2) get signout() from auth context

  async function load() {
    setLoading(true); setErr('')
    try {
      const res = await adminListUsers({ page, per_page: perPage, q })
      setItems(res.items); setTotal(res.total); setPages(res.pages || 1)
    } catch (e) {
      setErr(e?.data?.error||e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(()=>{ load() }, [page, perPage, q])

  async function createUser(data) {
    setSaving(true); setErr('')
    try {
      const payload = { ...data }
      if (!payload.password) payload.password = Math.random().toString(36).slice(2,10)
      await adminCreateUser(payload)
      setCreating(false)
      await load()
    } catch (e) {
      setErr(e?.data?.error||e.message)
    } finally {
      setSaving(false)
    }
  }

  async function saveEdit(data) {
    setSaving(true); setErr('')
    try {
      const id = editing.id
      const payload = { ...data }
      if (!payload.password) delete payload.password
      const res = await adminUpdateUser(id, payload)
      // ← (3a) if you demoted yourself, sign out right away
      if (res.self_demoted) {
        alert('Your admin rights were removed. You will be signed out.')
        signout()
        return
      }
      setEditing(null)
      await load()
    } catch (e) {
      setErr(e?.data?.error||e.message)
    } finally {
      setSaving(false)
    }
  }

  async function remove(id) {
    if (!confirm('Delete this user?')) return
    setErr('')
    try {
      const res = await adminDeleteUser(id)
      // ← (3b) if you deleted yourself, sign out right away
      if (res.self_deleted) {
        alert('Your admin account was deleted. You will be signed out.')
        signout()
        return
      }
      await load()
    } catch (e) {
      setErr(e?.data?.error||e.message)
    }
  }

  return (
    <div className="card">
      <h2 style={{marginTop:0}}>Admin · Users</h2>
      <div className="row" style={{justifyContent:'space-between', alignItems:'center'}}>
        <div className="row" style={{alignItems:'center'}}>
          <input placeholder="Search email/name/phone..." value={q} onChange={e=>{ setQ(e.target.value); setPage(1) }} style={{minWidth:280}} />
          <span className="help">Showing {items.length} of {total}</span>
        </div>
        <div className="row" style={{alignItems:'center'}}>
          <label>Per page</label>
          <select value={perPage} onChange={e=>{ setPerPage(parseInt(e.target.value)); setPage(1) }}>
            <option>10</option><option>20</option><option>50</option><option>100</option>
          </select>
          <button onClick={()=>setCreating(v=>!v)}>{creating ? 'Close' : 'New user'}</button>
        </div>
      </div>

      {err && <div className="help" style={{color:'#fecaca'}}>{err}</div>}
      {creating && (
        <div style={{margin:'12px 0'}}>
          <div className="section-title">Create user</div>
          <UserForm onSave={createUser} saving={saving} submitLabel="Create user" />
          <hr/>
        </div>
      )}

      <div style={{overflowX:'auto'}}>
        <table>
          <thead>
            <tr>
              <th>ID</th><th>Email</th><th>Name</th><th>Phone</th><th>City</th><th>Country</th><th>Credits</th><th>Admin</th><th></th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan="9">Loading...</td></tr>
            ) : items.length === 0 ? (
              <tr><td colSpan="9">No users</td></tr>
            ) : items.map(u => (
              <tr key={u.id}>
                <td>{u.id}</td>
                <td>{u.email}</td>
                <td>{[u.first_name,u.last_name].filter(Boolean).join(' ')}</td>
                <td>{u.phone||''}</td>
                <td>{u.city||''}</td>
                <td>{u.country||''}</td>
                <td>{u.credits}</td>
                <td>{u.is_admin ? 'Yes' : 'No'}</td>
                <td style={{textAlign:'right'}}>
                  <button className="secondary" onClick={()=>setEditing(u)}>Edit</button>{' '}
                  <button className="danger" onClick={()=>remove(u.id)}>Delete</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="row" style={{justifyContent:'space-between', marginTop:12}}>
        <div>Page {page} / {pages}</div>
        <div className="row">
          <button className="secondary" disabled={page<=1} onClick={()=>setPage(p=>p-1)}>Prev</button>
          <button disabled={page>=pages} onClick={()=>setPage(p=>p+1)} style={{marginLeft:8}}>Next</button>
        </div>
      </div>

      {editing && (
        <div style={{marginTop:16}}>
          <hr/>
          <div className="section-title">Edit user · #{editing.id}</div>
          <UserForm initial={editing} onSave={saveEdit} saving={saving} submitLabel="Save changes" />
          <div className="footer">Note: Leaving password blank keeps the current password.</div>
        </div>
      )}
    </div>
  )
}
