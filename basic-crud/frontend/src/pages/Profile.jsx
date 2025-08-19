import React, { useEffect, useState } from 'react'
import { getMe, updateMe, deleteMe } from '../api'
import { useAuth } from '../App'

export default function Profile() {
  const { setUser, signout } = useAuth()
  const [form, setForm] = useState({
    email: '', first_name:'', last_name:'', phone:'', address:'', city:'', country:'', password:''
  })
  const [saving, setSaving] = useState(false)
  const [err, setErr] = useState('')
  const [ok, setOk] = useState('')

  useEffect(() => {
    getMe().then(me => setForm(s => ({ ...s, ...me, password:'' }))).catch(e=>setErr(e?.data?.error||e.message))
  }, [])

  function setField(k, v){ setForm(s => ({...s, [k]: v})) }

  async function save(e){
    e.preventDefault()
    setErr(''); setOk(''); setSaving(true)
    const payload = { ...form }
    if (!payload.password) delete payload.password
    try {
      const res = await updateMe(payload)
      setOk('Saved')
      setUser(res.me)
    } catch (e) {
      setErr(e?.data?.error||e.message)
    } finally {
      setSaving(false)
    }
  }

  async function deleteAccount(){
    if (!confirm('Delete your account? This cannot be undone.')) return
    try {
      await deleteMe()
      alert('Account deleted')
      signout()
    } catch (e) {
      setErr(e?.data?.error||e.message)
    }
  }

  return (
    <div className="card">
      <h2 style={{marginTop:0}}>My Profile</h2>
      <form onSubmit={save}>
        <div className="row">
          <div className="field"><label>Email</label><input value={form.email||''} onChange={e=>setField('email', e.target.value)} /></div>
          <div className="field"><label>First name</label><input value={form.first_name||''} onChange={e=>setField('first_name', e.target.value)} /></div>
          <div className="field"><label>Last name</label><input value={form.last_name||''} onChange={e=>setField('last_name', e.target.value)} /></div>
          <div className="field"><label>Phone</label><input value={form.phone||''} onChange={e=>setField('phone', e.target.value)} /></div>
          <div className="field" style={{flex:1}}><label>Address</label><input value={form.address||''} onChange={e=>setField('address', e.target.value)} /></div>
          <div className="field"><label>City</label><input value={form.city||''} onChange={e=>setField('city', e.target.value)} /></div>
          <div className="field"><label>Country</label><input value={form.country||''} onChange={e=>setField('country', e.target.value)} /></div>
        </div>
        <div className="row">
          <div className="field"><label>Change password</label><input type="password" value={form.password||''} onChange={e=>setField('password', e.target.value)} placeholder="Leave blank to keep current" /></div>
        </div>
        {err && <div className="help" style={{color:'#fecaca'}}>{err}</div>}
        {ok && <div className="help" style={{color:'#bbf7d0'}}>{ok}</div>}
        <div className="row" style={{justifyContent:'space-between'}}>
          <button disabled={saving}>{saving?'Saving...':'Save changes'}</button>
          <button type="button" className="danger" onClick={deleteAccount}>Delete account</button>
        </div>
      </form>
    </div>
  )
}
