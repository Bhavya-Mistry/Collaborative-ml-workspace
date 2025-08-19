import React, { useEffect, useState } from 'react'
import { getDashboard } from '../api'

export default function Dashboard() {
  const [data, setData] = useState(null)
  const [err, setErr] = useState('')
  useEffect(() => {
    getDashboard().then(setData).catch(e=>setErr(e?.data?.error||e.message))
  }, [])

  if (err) return <div className="card"><div className="help" style={{color:'#fecaca'}}>{err}</div></div>
  if (!data) return <div className="card">Loading...</div>

  return (
    <div className="card">
      <h2 style={{marginTop:0}}>{data.title}</h2>
      <table>
        <thead><tr><th>Key</th><th>Value</th></tr></thead>
        <tbody>
          {data.rows.map((r, idx) => (
            <tr key={idx}>
              <td style={{width:220}}>{r.key}</td>
              <td>{String(r.value ?? '')}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
