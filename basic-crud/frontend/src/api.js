const BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:5000/api'

export function apiKey() {
  return localStorage.getItem('apiKey') || ''
}

export async function api(path, { method = 'GET', body, headers = {} } = {}) {
  const opts = { method, headers: { 'Content-Type': 'application/json', ...headers } }
  const key = apiKey()
  if (key) opts.headers['X-API-Key'] = key
  if (body !== undefined) opts.body = JSON.stringify(body)

  const res = await fetch(`${BASE}${path}`, opts)
  const text = await res.text()
  let data
  try { data = text ? JSON.parse(text) : {} } catch { data = { raw: text } }
  if (!res.ok) {
    const err = new Error(data?.error || 'Request failed')
    err.status = res.status
    err.data = data
    throw err
  }
  return data
}

export async function getMe() { return api('/me') }
export async function getDashboard() { return api('/me/dashboard') }
export async function signup(email, password) { return api('/signup', { method: 'POST', body: { email, password } }) }
export async function signin(email, password) { return api('/signin', { method: 'POST', body: { email, password } }) }
export async function updateMe(payload) { return api('/me', { method: 'PATCH', body: payload }) }
export async function deleteMe() { return api('/me', { method: 'DELETE' }) }

// Admin
export async function adminListUsers({ page = 1, per_page = 20, q = '' } = {}) {
  const qs = new URLSearchParams({ page, per_page, q }).toString()
  return api(`/admin/users?${qs}`)
}
export async function adminCreateUser(payload) { return api('/admin/users', { method: 'POST', body: payload }) }
export async function adminGetUser(id) { return api(`/admin/users/${id}`) }
export async function adminUpdateUser(id, payload) { return api(`/admin/users/${id}`, { method: 'PATCH', body: payload }) }
export async function adminDeleteUser(id) { return api(`/admin/users/${id}`, { method: 'DELETE' }) }
