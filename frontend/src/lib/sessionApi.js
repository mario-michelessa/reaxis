import { buildApiUrl } from './apiBase'

async function postJson(apiBase, path, body) {
  const res = await fetch(buildApiUrl(apiBase, path), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  })
  if (!res.ok) {
    let details = ''
    try { details = await res.text() } catch (_) {}
    throw new Error(`HTTP ${res.status} ${details}`.trim())
  }
  return res.json()
}

export async function openSession(apiBase, sessionName) {
  return postJson(apiBase, '/sessions/open', {
    session: String(sessionName || '').trim(),
  })
}

export async function appendSessionActivity(apiBase, sessionName, action, detail) {
  if (!String(sessionName || '').trim()) return null
  return postJson(apiBase, '/session/log', {
    session: String(sessionName || '').trim(),
    action: String(action || '').trim(),
    detail: String(detail || '').trim(),
  })
}
