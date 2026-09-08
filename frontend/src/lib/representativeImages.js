const DEFAULT_REPRESENTATIVE_COUNT = 10

function normalizeId(value) {
  return String(value || '').trim()
}

function compareById(a, b) {
  return String(a?.id || '').localeCompare(String(b?.id || ''))
}

function squaredDistance(a, b) {
  const dx = Number(a?.x || 0) - Number(b?.x || 0)
  const dy = Number(a?.y || 0) - Number(b?.y || 0)
  return (dx * dx) + (dy * dy)
}

function normalizeCandidates(items) {
  const candidates = []
  for (const raw of Array.isArray(items) ? items : []) {
    const id = normalizeId(raw?.id)
    if (!id) continue
    const x = Number.isFinite(Number(raw?.x)) ? Number(raw.x) : Number(raw?.gx)
    const y = Number.isFinite(Number(raw?.y)) ? Number(raw.y) : Number(raw?.gy)
    candidates.push({
      id,
      x,
      y,
      hasCoords: Number.isFinite(x) && Number.isFinite(y),
    })
  }
  candidates.sort(compareById)
  return candidates
}

function evenlySpacedIds(candidates, count) {
  const uniqueIds = []
  const seen = new Set()
  for (const candidate of candidates) {
    const id = normalizeId(candidate?.id)
    if (!id || seen.has(id)) continue
    seen.add(id)
    uniqueIds.push(id)
  }
  if (uniqueIds.length <= count) return uniqueIds

  const picks = []
  const used = new Set()
  const step = (uniqueIds.length - 1) / Math.max(1, count - 1)
  for (let i = 0; i < count; i += 1) {
    const target = Math.round(i * step)
    let offset = 0
    while (offset < uniqueIds.length) {
      const forward = target + offset
      if (forward < uniqueIds.length && !used.has(uniqueIds[forward])) {
        used.add(uniqueIds[forward])
        picks.push(uniqueIds[forward])
        break
      }
      if (offset > 0) {
        const backward = target - offset
        if (backward >= 0 && !used.has(uniqueIds[backward])) {
          used.add(uniqueIds[backward])
          picks.push(uniqueIds[backward])
          break
        }
      }
      offset += 1
    }
  }
  return picks
}

export function selectRepresentativeImageIds(items, count = DEFAULT_REPRESENTATIVE_COUNT) {
  const targetCount = Math.max(1, Number(count) || DEFAULT_REPRESENTATIVE_COUNT)
  const candidates = normalizeCandidates(items)
  if (candidates.length === 0) return []
  if (candidates.length <= targetCount) return candidates.map((candidate) => candidate.id)

  const withCoords = candidates.filter((candidate) => candidate.hasCoords)
  if (withCoords.length < 2) return evenlySpacedIds(candidates, targetCount)

  let centroidX = 0
  let centroidY = 0
  for (const candidate of withCoords) {
    centroidX += candidate.x
    centroidY += candidate.y
  }
  centroidX /= withCoords.length
  centroidY /= withCoords.length

  let first = withCoords[0]
  let firstDistance = -1
  for (const candidate of withCoords) {
    const distance = squaredDistance(candidate, { x: centroidX, y: centroidY })
    if (distance > firstDistance + 1e-9) {
      first = candidate
      firstDistance = distance
      continue
    }
    if (Math.abs(distance - firstDistance) <= 1e-9 && compareById(candidate, first) < 0) {
      first = candidate
    }
  }

  const selected = [first]
  const selectedIds = new Set([first.id])
  while (selected.length < Math.min(targetCount, withCoords.length)) {
    let best = null
    let bestMinDistance = -1
    for (const candidate of withCoords) {
      if (selectedIds.has(candidate.id)) continue
      let minDistance = Number.POSITIVE_INFINITY
      for (const picked of selected) {
        const distance = squaredDistance(candidate, picked)
        if (distance < minDistance) minDistance = distance
      }
      if (minDistance > bestMinDistance + 1e-9) {
        best = candidate
        bestMinDistance = minDistance
        continue
      }
      if (
        best &&
        Math.abs(minDistance - bestMinDistance) <= 1e-9 &&
        compareById(candidate, best) < 0
      ) {
        best = candidate
      }
    }
    if (!best) break
    selected.push(best)
    selectedIds.add(best.id)
  }

  if (selected.length < targetCount) {
    for (const id of evenlySpacedIds(candidates, targetCount)) {
      if (selectedIds.has(id)) continue
      selected.push({ id })
      selectedIds.add(id)
      if (selected.length >= targetCount) break
    }
  }

  return selected.slice(0, targetCount).map((candidate) => candidate.id)
}
