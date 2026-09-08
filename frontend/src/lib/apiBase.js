const TRAILING_SLASH_RE = /\/+$/

function normalizeApiBase(value) {
  const base = String(value ?? '').trim()
  if (!base) return ''
  return base.replace(TRAILING_SLASH_RE, '')
}

export function resolveApiBase() {
  const fromEnv = normalizeApiBase(import.meta.env?.VITE_API_BASE)
  if (fromEnv) return fromEnv
  return '/api'
}

export function buildApiUrl(base, path) {
  const normalizedBase = normalizeApiBase(base)
  const normalizedPath = String(path ?? '').trim()
  if (!normalizedBase) return normalizedPath
  if (!normalizedPath) return normalizedBase
  if (normalizedPath.startsWith('/')) return `${normalizedBase}${normalizedPath}`
  return `${normalizedBase}/${normalizedPath}`
}
