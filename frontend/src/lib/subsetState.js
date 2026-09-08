function clamp01(v) {
  const n = Number(v)
  if (!Number.isFinite(n)) return 0
  if (n <= 0) return 0
  if (n >= 1) return 1
  return n
}

function clamp100(v) {
  return clamp01((Number(v) || 0) / 100) * 100
}

export function normalizeIdList(raw) {
  if (!Array.isArray(raw)) return []
  const out = []
  const seen = new Set()
  for (const entry of raw) {
    const id = String(entry || '').trim()
    if (!id || seen.has(id)) continue
    seen.add(id)
    out.push(id)
  }
  return out
}

export function normalizeHistogramSlice(raw) {
  if (!raw || typeof raw !== 'object') return null
  const axisId = String(raw.axisId || '').trim()
  if (!axisId) return null
  const ids = normalizeIdList(raw.ids)
  return {
    axisId,
    axisName: String(raw.axisName || raw.axisLabel || axisId).trim() || axisId,
    startBin: Number(raw.startBin || 0),
    endBin: Number(raw.endBin || 0),
    binCount: Number(raw.binCount || 0),
    rangeStartPct: clamp100(raw.rangeStartPct || 0),
    rangeEndPct: clamp100(raw.rangeEndPct || 0),
    ids,
  }
}

export function normalizeHistogramSlices(raw) {
  if (!Array.isArray(raw)) return []
  const out = []
  const seen = new Set()
  for (const entry of raw) {
    const slice = normalizeHistogramSlice(entry)
    if (!slice || seen.has(slice.axisId)) continue
    seen.add(slice.axisId)
    out.push(slice)
  }
  return out
}

function subsetFilterId(prefix = 'subset') {
  return `${prefix}:${Date.now()}:${Math.random().toString(36).slice(2, 8)}`
}

export function subsetFilterLabel(raw) {
  const mode = String(raw?.mode || '').trim().toLowerCase() === 'exclude' ? 'exclude' : 'isolate'
  const count = normalizeIdList(raw?.ids).length
  return `${mode === 'exclude' ? 'Exclude' : 'Isolate'} (${count})`
}

export function normalizeSubsetFilter(raw) {
  if (!raw || typeof raw !== 'object') return null
  const ids = normalizeIdList(raw.ids)
  if (ids.length === 0) return null
  const mode = String(raw.mode || '').trim().toLowerCase() === 'exclude' ? 'exclude' : 'isolate'
  const id = String(raw.id || '').trim() || subsetFilterId(`subset:${mode}`)
  return {
    id,
    mode,
    ids,
  }
}

export function normalizeSubsetFilters(raw) {
  if (Array.isArray(raw)) {
    return raw
      .map((entry) => normalizeSubsetFilter(entry))
      .filter(Boolean)
  }
  const single = normalizeSubsetFilter(raw)
  return single ? [single] : []
}

export function createSubsetFilter(mode, ids) {
  const normalizedIds = normalizeIdList(ids)
  if (normalizedIds.length === 0) return null
  return {
    id: subsetFilterId(`subset:${String(mode || 'isolate').trim().toLowerCase() === 'exclude' ? 'exclude' : 'isolate'}`),
    mode: String(mode || '').trim().toLowerCase() === 'exclude' ? 'exclude' : 'isolate',
    ids: normalizedIds,
  }
}

export function sliceChipLabel(raw) {
  const slice = normalizeHistogramSlice(raw)
  if (!slice) return 'Slice'
  return `Slice: ${slice.axisName || slice.axisId || 'Axis'}`
}

export function subsetChipsFromState(histogramSlices, subsetFilters) {
  const sliceChips = normalizeHistogramSlices(histogramSlices).map((slice) => ({
    id: `slice:${slice.axisId}`,
    chipType: 'slice',
    label: sliceChipLabel(slice),
    ...slice,
  }))
  const filterChips = normalizeSubsetFilters(subsetFilters).map((filter) => ({
    id: filter.id,
    chipType: 'filter',
    label: subsetFilterLabel(filter),
    ...filter,
  }))
  return [...sliceChips, ...filterChips]
}

export function stateFromSubsetChips(raw) {
  const chips = Array.isArray(raw) ? raw : []
  const histogramSlices = []
  const subsetFilters = []
  for (const chip of chips) {
    if (!chip || typeof chip !== 'object') continue
    const chipType = String(chip.chipType || chip.type || chip.kind || '').trim().toLowerCase()
    if (chipType === 'slice') {
      const slice = normalizeHistogramSlice(chip)
      if (slice) histogramSlices.push(slice)
      continue
    }
    const filter = normalizeSubsetFilter(chip)
    if (filter) subsetFilters.push(filter)
  }
  return {
    histogramSlices: normalizeHistogramSlices(histogramSlices),
    subsetFilters: normalizeSubsetFilters(subsetFilters),
  }
}

export function hasActiveSubset(histogramSlices, subsetFilters) {
  return normalizeHistogramSlices(histogramSlices).length > 0 || normalizeSubsetFilters(subsetFilters).length > 0
}

export function deriveEffectiveSubsetIds(items, histogramSlices, subsetFilters) {
  const allIds = (Array.isArray(items) ? items : [])
    .map((item) => String(item?.id || '').trim())
    .filter(Boolean)
  const activeSlices = normalizeHistogramSlices(histogramSlices)
  const activeFilters = normalizeSubsetFilters(subsetFilters)
  if (activeSlices.length === 0 && activeFilters.length === 0) return allIds

  let keep = new Set(allIds)
  for (const slice of activeSlices) {
    const ids = new Set(slice.ids)
    keep = new Set(Array.from(keep).filter((id) => ids.has(id)))
  }
  for (const filter of activeFilters) {
    const ids = new Set(filter.ids)
    if (filter.mode === 'exclude') {
      keep = new Set(Array.from(keep).filter((id) => !ids.has(id)))
    } else {
      keep = new Set(Array.from(keep).filter((id) => ids.has(id)))
    }
  }
  return allIds.filter((id) => keep.has(id))
}
