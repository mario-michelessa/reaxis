export const METADATA_AXIS_PALETTE = [
  '#2563eb',
  '#7c3aed',
  '#db2777',
  '#ea580c',
  '#ca8a04',
  '#16a34a',
  '#0891b2',
  '#4f46e5',
  '#c026d3',
  '#dc2626',
  '#65a30d',
  '#0f766e',
  '#9333ea',
  '#e11d48',
  '#0284c7',
  '#475569',
]

export function isMetadataAxis(axis) {
  if (!axis || typeof axis !== 'object') return false
  const group = String(axis.group || '').trim().toLowerCase()
  const id = String(axis.id || '').trim()
  return group === 'meta' || id.startsWith('axis:meta:')
}

function hexToRgb(hex) {
  const raw = String(hex || '').trim().replace('#', '')
  const norm = raw.length === 3
    ? raw.split('').map((ch) => ch + ch).join('')
    : raw
  if (!/^[0-9a-fA-F]{6}$/.test(norm)) return null
  return {
    r: parseInt(norm.slice(0, 2), 16),
    g: parseInt(norm.slice(2, 4), 16),
    b: parseInt(norm.slice(4, 6), 16),
  }
}

export function rgbaFromHex(hex, alpha = 1) {
  const rgb = hexToRgb(hex)
  const a = Math.max(0, Math.min(1, Number(alpha) || 0))
  if (!rgb) return `rgba(148,163,184,${a})`
  return `rgba(${rgb.r},${rgb.g},${rgb.b},${a})`
}

export function buildMetadataAxisEntries(axis) {
  const labels = Array.isArray(axis?.labels) ? axis.labels : []
  const positions = Array.isArray(axis?.label_positions) ? axis.label_positions : []
  const limit = Math.min(labels.length, positions.length)
  const entries = []
  const axisId = String(axis?.id || 'axis')
  for (let idx = 0; idx < limit; idx += 1) {
    const pos = Number(positions[idx])
    if (!Number.isFinite(pos)) continue
    const paletteIndex = entries.length % METADATA_AXIS_PALETTE.length
    entries.push({
      key: `${axisId}::${idx}::${String(labels[idx] || '')}`,
      label: String(labels[idx] || '').trim() || `Label ${idx + 1}`,
      pos,
      index: entries.length,
      paletteIndex,
      color: METADATA_AXIS_PALETTE[paletteIndex],
    })
  }
  entries.sort((a, b) => a.pos - b.pos || a.index - b.index)
  return entries.map((entry, idx) => ({
    ...entry,
    index: idx,
    paletteIndex: idx % METADATA_AXIS_PALETTE.length,
    color: METADATA_AXIS_PALETTE[idx % METADATA_AXIS_PALETTE.length],
  }))
}

function nearestMetadataEntry(entries, value) {
  let best = null
  let bestDist = Infinity
  for (const entry of entries) {
    const dist = Math.abs(Number(value) - Number(entry.pos))
    if (dist < bestDist) {
      best = entry
      bestDist = dist
    }
  }
  return best
}

export function buildMetadataColorLookup(axis) {
  const lookup = new Map()
  const entries = buildMetadataAxisEntries(axis)
  if (!entries.length) return lookup
  const coords = axis?.coords && typeof axis.coords === 'object' ? axis.coords : {}
  for (const [rawId, rawValue] of Object.entries(coords)) {
    const id = String(rawId || '').trim()
    const value = Number(rawValue)
    if (!id || !Number.isFinite(value)) continue
    const entry = nearestMetadataEntry(entries, value)
    if (!entry) continue
    lookup.set(id, {
      axisId: String(axis?.id || ''),
      label: entry.label,
      pos: entry.pos,
      value,
      paletteIndex: entry.paletteIndex,
      color: entry.color,
      pointColor: rgbaFromHex(entry.color, 0.84),
      pointColorStrong: rgbaFromHex(entry.color, 0.96),
      pointColorMuted: rgbaFromHex(entry.color, 0.74),
      outlineColor: entry.color,
      outlineSoft: rgbaFromHex(entry.color, 0.18),
    })
  }
  return lookup
}
