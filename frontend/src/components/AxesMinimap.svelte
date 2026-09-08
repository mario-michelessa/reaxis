<script>
  // AxesMinimap: like HoverGridMinimap, but x/y come from selected 1D axes.
  // - Drag axes from AxesPanel into X/Y dropzones to change current axes
  // - Uses local packing inside selection window for image grid placement

  import { createEventDispatcher, onMount, onDestroy } from 'svelte'
  import { axisBuildersStore } from '../lib/axisBuilderStore'
  import { buildApiUrl, resolveApiBase } from '../lib/apiBase'
  import { axisPayloadFromSession, axisSessionFromResponse } from '../lib/axisSessions'
  import {
    buildMetadataAxisEntries,
    buildMetadataColorLookup,
    isMetadataAxis,
  } from '../lib/metadataAxisColoring'
  import {
    createSubsetFilter,
    normalizeSubsetFilters,
    subsetChipsFromState,
  } from '../lib/subsetState'
  import LassoSelector from './LassoSelector.svelte'
  import MaterialIcon from './MaterialIcon.svelte'
  export let items = [] // [{ id, url, thumbUrl?, fullUrl?, x, y, gx, gy }]
  export let axes = [] // [{ id, name, coords: Record<string, number> }]
  export let apiBase = resolveApiBase()
  export let sessionName = ''
  export let width = 700
  export let height = 700
  export let viewFrac = 0.15 // viewport square fraction (initial)
  export let duration = 300 // ms
  export let minImagePx = 25
  export let posUpdateMs = 450
  export let imageMax = 400
  export let imageSubsampleSeed = 1337
  export let focusImageRequest = null
  export let restoreViewState = null
  export let subsetFilters = []
  export let saveVisualizationDisabled = false
  const MIN_CELL_PX = 10
  const MAX_CELL_PX = 32
  const CELL_PX_STEP = 2
  const Y_AXIS_EDGE_NUDGE_PX = 90
  let cellPx = 16

  const dispatch = createEventDispatcher()
  $: axisBuilderSessions = $axisBuildersStore
  function emitLog(action, detail = 'none') {
    const actionText = String(action || '').trim()
    if (!actionText) return
    dispatch('logAction', {
      action: actionText,
      detail: String(detail || '').trim() || 'none',
    })
  }
  function touch(..._args) {}

  // Visual margin for display (map [0,1] -> [m, 1-m])
  export let displayMargin = 0.1
  function toDisplayUnit(v) {
    const m = Math.max(0, Math.min(0.49, Number(displayMargin || 0)))
    const nv = Number(v || 0)
    return m + nv * (1 - 2 * m)
  }
  function fromDisplayUnit(v) {
    const m = Math.max(0, Math.min(0.49, Number(displayMargin || 0)))
    const nv = Number(v || 0)
    const denom = Math.max(1e-6, (1 - 2 * m))
    return (nv - m) / denom
  }
  $: squareSidePx = Math.max(1, Math.min(Number(width || 1), Number(height || 1)))
  $: squareWNorm = squareSidePx / Math.max(1, Number(width || 1))
  $: squareHNorm = squareSidePx / Math.max(1, Number(height || 1))
  $: squareXOffsetNorm = (1 - squareWNorm) * 0.5
  $: squareYOffsetNorm = (1 - squareHNorm) * 0.5
  function toVisX(v) {
    return squareXOffsetNorm + toDisplayUnit(v) * squareWNorm
  }
  function toVisY(v) {
    return squareYOffsetNorm + toDisplayUnit(v) * squareHNorm
  }
  function fromVisX(v) {
    const local = (Number(v || 0) - squareXOffsetNorm) / Math.max(1e-6, squareWNorm)
    return fromDisplayUnit(local)
  }
  function fromVisY(v) {
    const local = (Number(v || 0) - squareYOffsetNorm) / Math.max(1e-6, squareHNorm)
    return fromDisplayUnit(local)
  }
  function screenNormToWorldX(v) {
    const zx = fromVisX(v)
    const wx = cx + (zx - cx) / Math.max(1e-6, zoomZ)
    return clamp(wx, 0, 1)
  }
  function screenNormToWorldYTop(v) {
    const zyBottom = fromVisY(1 - Number(v || 0))
    const wyBottom = cy + (zyBottom - cy) / Math.max(1e-6, zoomZ)
    return clamp(1 - wyBottom, 0, 1)
  }
  function worldXToScreenNorm(v) {
    const wx = Number(v || 0)
    return toVisX(cx + (wx - cx) * zoomZ)
  }
  function worldYTopToScreenNorm(v) {
    const topY = Number(v || 0)
    const wyBottom = 1 - topY
    const zyBottom = cy + (wyBottom - cy) * zoomZ
    return 1 - toVisY(zyBottom)
  }

  // External labels for outline rendering and axis creation
  export let labels = new Map()
  function labelOf(id) {
    try {
      if (!labels) return undefined
      if (labels instanceof Map) return labels.get(id)
      return labels[id]
    } catch (_) { return undefined }
  }

  // Counts for labeled items
  $: posCount = (() => {
    let c = 0
    if (!labels) return 0
    try {
      if (labels instanceof Map) { labels.forEach((v) => { if (v === 'pos') c++ }) }
      else { for (const v of Object.values(labels)) if (v === 'pos') c++ }
    } catch (_) {}
    return c
  })()
  $: negCount = (() => {
    let c = 0
    if (!labels) return 0
    try {
      if (labels instanceof Map) { labels.forEach((v) => { if (v === 'neg') c++ }) }
      else { for (const v of Object.values(labels)) if (v === 'neg') c++ }
    } catch (_) {}
    return c
  })()

  // Lasso subset selection
  let lassoEnabled = false
  let subsetSelectionIds = []
  let lassoRef
  $: subsetSelectionSet = new Set((subsetSelectionIds || []).map((id) => String(id || '').trim()).filter(Boolean))
  $: activeSubsetFilters = normalizeSubsetFilters(subsetFilters)
  function clearLassoSelection() {
    subsetSelectionIds = []
    try { lassoRef && lassoRef.reset && lassoRef.reset() } catch (_) {}
  }
  function onLassoSelect(e) {
    const ids = Array.isArray(e.detail && e.detail.ids) ? e.detail.ids.map((id) => String(id || '').trim()).filter(Boolean) : []
    subsetSelectionIds = ids
    if (!ids || ids.length === 0) {
      try { lassoRef && lassoRef.reset && lassoRef.reset() } catch (_) {}
      return
    }
    suppressUntil = performance.now() + 400
  }
  function applySubsetFilter(mode) {
    const ids = Array.from(subsetSelectionSet)
    if (!ids.length) return
    const nextFilter = createSubsetFilter(mode, ids)
    if (!nextFilter) return
    dispatch('subsetFiltersChange', { filters: [...activeSubsetFilters, nextFilter] })
    clearLassoSelection()
    emitLog('lasso selection', `${nextFilter.mode} | ${ids.length}`)
  }
  function removeSubsetFilter(filterId, detail = 'none') {
    const id = String(filterId || '').trim()
    if (!id) return
    const next = activeSubsetFilters.filter((entry) => String(entry?.id || '').trim() !== id)
    dispatch('subsetFiltersChange', { filters: next })
    emitLog('clear subset', detail)
  }
  function removeHistogramSlice(axisId, detail = 'none') {
    const id = String(axisId || '').trim()
    if (!id) return
    const next = activeHistogramSlices.filter((entry) => String(entry?.axisId || '').trim() !== id)
    dispatch('sliceChange', { slices: next, slice: next.length === 1 ? next[0] : null })
    emitLog('clear subset', detail)
  }
  function visibleScatterPositionForItem(it) {
    const pos = screenPositionForItem(it)
    if (!Number.isFinite(pos?.x) || !Number.isFinite(pos?.y)) return null
    if (pos.x < 0 || pos.x > 1 || pos.y < 0 || pos.y > 1) return null
    return pos
  }
  function toggleLassoTool() {
    lassoEnabled = !lassoEnabled
    grabMode = false
    grabDrag = null
    stopGrabListeners()
    if (!lassoEnabled) clearLassoSelection()
  }
  function originalVector(it) {
    if (Array.isArray(it?.originalEmbed) && it.originalEmbed.length > 0) {
      const out = it.originalEmbed.map((v) => Number(v)).filter((v) => Number.isFinite(v))
      if (out.length > 0) return out
    }
    if (Array.isArray(it?.embed) && it.embed.length > 0) {
      const out = it.embed.map((v) => Number(v)).filter((v) => Number.isFinite(v))
      if (out.length > 0) return out
    }
    return [Number(it?.x ?? 0), Number(it?.y ?? 0)]
  }
  function vecDist(a, b) {
    const va = originalVector(a)
    const vb = originalVector(b)
    const n = Math.max(va.length, vb.length)
    let sum = 0
    for (let i = 0; i < n; i++) {
      const da = (Number(va[i]) || 0) - (Number(vb[i]) || 0)
      sum += da * da
    }
    return Math.sqrt(sum)
  }
  function createAxisFromLabels() {
    // Build axis scores based on current labels
    const posIds = new Set()
    const negIds = new Set()
    if (labels) {
      if (labels instanceof Map) {
        labels.forEach((v, k) => { if (v === 'pos') posIds.add(k); if (v === 'neg') negIds.add(k) })
      } else {
        for (const [k, v] of Object.entries(labels)) { if (v === 'pos') posIds.add(k); if (v === 'neg') negIds.add(k) }
      }
    }
    const posItems = items.filter(i => posIds.has(i.id))
    const negItems = items.filter(i => negIds.has(i.id))
    // Distance is computed in the original feature space, not the active projected axes.
    function minDist(pt, arr) { if (!arr.length) return Infinity; let m = Infinity; for (const s of arr) { const d = vecDist(pt, s); if (d < m) m = d } return m }
    const coords = {}
    let maxDp = 0, maxDn = 0
    for (const it of items) { const dp = minDist(it, posItems); const dn = minDist(it, negItems); if (isFinite(dp) && dp > maxDp) maxDp = dp; if (isFinite(dn) && dn > maxDn) maxDn = dn }
    function clamp01(v) { return Math.max(0, Math.min(1, v)) }
    for (const it of items) {
      const dp = minDist(it, posItems)
      const dn = minDist(it, negItems)
      let s = 0.5
      if (posItems.length && negItems.length) s = clamp01(dn / (dn + dp + 1e-9))
      else if (posItems.length && !negItems.length) s = clamp01(1 - (dp / (maxDp + 1e-9)))
      else if (negItems.length && !posItems.length) s = clamp01(dn / (maxDn + 1e-9))
      coords[it.id] = s
    }
    const name = prompt('Name this axis', 'Axis') || 'Axis'
    const id = `axis:labels:${Date.now()}`
    dispatch('create', { id, name, coords })
  }

  // Selected axis ids for X and Y
  export let selectedX = null
  export let selectedY = null
  let selectedColorAxisId = null

  $: axesById = new Map((axes || []).map(a => [a.id, a]))
  $: metadataAxes = (axes || []).filter((axis) => isMetadataAxis(axis))
  $: metadataAxesById = new Map((metadataAxes || []).map((axis) => [axis.id, axis]))
  $: selectedXAxis = selectedX ? (axesById.get(selectedX) || null) : null
  $: selectedYAxis = selectedY ? (axesById.get(selectedY) || null) : null
  $: selectedColorAxis = selectedColorAxisId ? (metadataAxesById.get(selectedColorAxisId) || null) : null
  $: metadataColorLookup = buildMetadataColorLookup(selectedColorAxis)
  $: if (selectedColorAxisId && !metadataAxesById.has(selectedColorAxisId)) selectedColorAxisId = null
  $: scatterLayoutDeps = {
    selectedX,
    selectedY,
    selectedXAxis,
    selectedYAxis,
    selectedXAxisCoords: selectedXAxis?.coords || null,
    selectedYAxisCoords: selectedYAxis?.coords || null,
    selectedColorAxisId,
    selectedColorAxis,
    selectedColorAxisCoords: selectedColorAxis?.coords || null,
    sameAxisKdeLayout,
    showUncertainty,
  }
  $: itemsById = new Map((items || []).map((item) => [String(item?.id || ''), item]).filter((row) => row[0]))

  function axisName(id) { return (axesById.get(id)?.name) || '—' }
  $: axisFrameCenterXPx = (squareXOffsetNorm * width) + (squareSidePx * 0.5)
  $: axisFrameCenterYPx = (squareYOffsetNorm * height) + (squareSidePx * 0.5)
  $: axisFrameLeftPx = squareXOffsetNorm * width
  $: axisFrameBottomPx = (squareYOffsetNorm * height) + squareSidePx
  $: axisFrameTopPct = squareYOffsetNorm * 100
  $: axisFrameLeftPct = squareXOffsetNorm * 100
  $: axisFrameWidthPct = squareWNorm * 100
  $: axisFrameHeightPct = squareHNorm * 100
  function getCoord(id, axisId) {
    const ax = axisId ? axesById.get(axisId) : null
    if (!ax || !ax.coords) return undefined
    const v = ax.coords[id]
    return (typeof v === 'number' && isFinite(v)) ? Math.max(0, Math.min(1, v)) : undefined
  }
  function axisTicks(axisId) {
    const ax = axisId ? axesById.get(axisId) : null
    if (!ax) return { labels: [], pos: [], entries: [] }
    const entries = buildMetadataAxisEntries(ax)
    const labels = entries.map((entry) => entry.label)
    const pos = entries.map((entry) => entry.pos)
    return { labels, pos, entries }
  }
  function normalizeAxisType(v) {
    const s = String(v || '').trim().toLowerCase()
    if (s === 'continuous' || s === 'ordinal' || s === 'categorical') return s
    if (s.includes('contin') || s.includes('numeric') || s.includes('scalar')) return 'continuous'
    if (s.includes('ordin') || s.includes('rank')) return 'ordinal'
    if (s.includes('categor') || s.includes('nominal') || s.includes('class')) return 'categorical'
    return ''
  }
  function inferNonContinuousAxis(ax) {
    if (!ax) return false
    const t = normalizeAxisType(ax.attribute_type || ax.type || ax.axis_type)
    if (t === 'continuous') return false
    if (t === 'categorical' || t === 'ordinal') return true
    const method = String(ax.scoring_method || '').trim().toLowerCase()
    if (method.includes('classification')) return true
    if (method.includes('direction_projection')) return false

    const labels = Array.isArray(ax.labels) ? ax.labels : []
    const positions = Array.isArray(ax.label_positions) ? ax.label_positions : []
    if (labels.length >= 2 && labels.length <= 24 && labels.length === positions.length) return true
    if (String(ax.group || '').toLowerCase() === 'meta' && labels.length > 0 && labels.length <= 32) return true

    const coords = ax.coords || {}
    const seen = new Set()
    let sampled = 0
    for (const v of Object.values(coords)) {
      const n = Number(v)
      if (!Number.isFinite(n)) continue
      seen.add(Math.round(n * 1000))
      sampled += 1
      if (sampled >= 600 || seen.size > 48) break
    }
    if (sampled > 0 && seen.size <= Math.max(6, Math.round(Math.sqrt(sampled)))) return true
    return false
  }
  let nonContinuousAxisById = new Map()
  $: nonContinuousAxisById = (() => {
    const next = new Map()
    for (const ax of (axes || [])) next.set(ax.id, inferNonContinuousAxis(ax))
    return next
  })()
  function isNonContinuousAxis(axisId) {
    if (!axisId) return false
    return !!nonContinuousAxisById.get(axisId)
  }
  function apiUrl(path) {
    return buildApiUrl(apiBase, path)
  }
  async function postJson(path, body) {
    const res = await fetch(apiUrl(path), {
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
  function findBuilder(axisId) {
    return (axisBuilderSessions || []).find((entry) => entry?.axisId === axisId) || null
  }
  function normalizeImageId(v) {
    return String(v || '').trim()
  }
  function basenameId(v) {
    const s = normalizeImageId(v)
    if (!s) return ''
    const parts = s.split(/[\\/]/)
    return parts[parts.length - 1] || s
  }
  function sameImageId(a, b) {
    const aa = normalizeImageId(a)
    const bb = normalizeImageId(b)
    if (!aa || !bb) return false
    return aa === bb || basenameId(aa) === basenameId(bb)
  }
  function resolveExistingImageId(rawImageId) {
    const target = normalizeImageId(rawImageId)
    if (!target) return ''
    if (itemsById.has(target)) return target
    const base = basenameId(target)
    if (base && itemsById.has(base)) return base
    const keys = Array.from(itemsById.keys())
    const matches = keys.filter((id) => sameImageId(id, target))
    if (matches.length === 1) return String(matches[0])
    if (matches.length > 1) return String(matches[0])
    return ''
  }
  function scoreForBuilderImage(builder, imageId) {
    if (!builder || !Array.isArray(builder.ids) || !Array.isArray(builder.scores)) return null
    const idx = builder.ids.findIndex((id) => sameImageId(id, imageId))
    if (idx < 0 || idx >= builder.scores.length) return null
    const value = Number(builder.scores[idx] || 0)
    return Number.isFinite(value) ? value : null
  }
  function resolveBuilderImageId(builder, imageId) {
    const raw = normalizeImageId(imageId)
    if (!raw || !builder || !Array.isArray(builder.ids)) return raw
    const exact = builder.ids.find((id) => normalizeImageId(id) === raw)
    if (exact) return String(exact)
    const base = basenameId(raw)
    if (!base) return raw
    const byBase = builder.ids.filter((id) => basenameId(id) === base)
    if (byBase.length === 1) return String(byBase[0])
    return raw
  }
  let showUncertainty = false
  $: axisStdStats = (() => {
    const out = new Map()
    const sessions = Array.isArray(axisBuilderSessions) ? axisBuilderSessions : []
    for (const builder of sessions) {
      const axisId = String(builder?.axisId || builder?.axis?.id || '').trim()
      if (!axisId) continue
      const ids = Array.isArray(builder?.ids) ? builder.ids : []
      const std = Array.isArray(builder?.std) ? builder.std : []
      const n = Math.min(ids.length, std.length)
      if (n <= 0) continue
      const byExact = new Map()
      const byBaseCount = new Map()
      let min = Infinity
      let max = -Infinity
      for (let i = 0; i < n; i += 1) {
        const id = normalizeImageId(ids[i])
        const s = Number(std[i])
        if (!id || !Number.isFinite(s)) continue
        byExact.set(id, s)
        const b = basenameId(id)
        if (b) byBaseCount.set(b, (byBaseCount.get(b) || 0) + 1)
        if (s < min) min = s
        if (s > max) max = s
      }
      const byBaseUnique = new Map()
      for (const [id, s] of byExact.entries()) {
        const b = basenameId(id)
        if (!b) continue
        if ((byBaseCount.get(b) || 0) === 1) byBaseUnique.set(b, s)
      }
      out.set(axisId, { byExact, byBaseUnique, min: Number.isFinite(min) ? min : 0, max: Number.isFinite(max) ? max : 1 })
    }
    return out
  })()
  $: imageUncertaintyGlobal = (() => {
    const byExact = new Map()
    const byBaseAccum = new Map()
    for (const stats of axisStdStats.values()) {
      const span = Math.max(1e-8, Number(stats.max) - Number(stats.min))
      for (const [id, sRaw] of stats.byExact.entries()) {
        const s = Number(sRaw)
        if (!Number.isFinite(s)) continue
        const norm = Math.max(0, Math.min(1, (s - Number(stats.min)) / span))
        const key = normalizeImageId(id)
        if (!key) continue
        const prev = byExact.get(key) || { sum: 0, count: 0 }
        byExact.set(key, { sum: prev.sum + norm, count: prev.count + 1 })
        const b = basenameId(key)
        if (b) {
          const prevB = byBaseAccum.get(b) || { sum: 0, count: 0 }
          byBaseAccum.set(b, { sum: prevB.sum + norm, count: prevB.count + 1 })
        }
      }
    }
    const outExact = new Map()
    for (const [k, v] of byExact.entries()) outExact.set(k, v.sum / Math.max(1, v.count))
    const outBase = new Map()
    for (const [k, v] of byBaseAccum.entries()) outBase.set(k, v.sum / Math.max(1, v.count))
    return { byExact: outExact, byBase: outBase }
  })()
  function stdForAxisImage(axisId, imageId) {
    const axisKey = String(axisId || '').trim()
    const id = normalizeImageId(imageId)
    if (!axisKey || !id || !axisStdStats.has(axisKey)) return null
    const stats = axisStdStats.get(axisKey)
    if (stats.byExact.has(id)) return Number(stats.byExact.get(id))
    const b = basenameId(id)
    if (b && stats.byBaseUnique.has(b)) return Number(stats.byBaseUnique.get(b))
    return null
  }
  function uncertaintyNormForImage(imageId) {
    if (!showUncertainty) return null
    const preferred = [selectedX, selectedY]
      .map((axisId) => String(axisId || '').trim())
      .filter((axisId) => axisStdStats.has(axisId))
    const axisIds = preferred.length > 0 ? preferred : Array.from(axisStdStats.keys())
    if (axisIds.length === 0) return null
    let sum = 0
    let count = 0
    for (const axisId of axisIds) {
      const stats = axisStdStats.get(String(axisId || ''))
      if (!stats) continue
      const s = stdForAxisImage(axisId, imageId)
      if (!Number.isFinite(s)) continue
      const span = Math.max(1e-8, Number(stats.max) - Number(stats.min))
      const norm = Math.max(0, Math.min(1, (Number(s) - Number(stats.min)) / span))
      sum += norm
      count += 1
    }
    if (count <= 0) {
      const id = normalizeImageId(imageId)
      if (imageUncertaintyGlobal.byExact.has(id)) return Number(imageUncertaintyGlobal.byExact.get(id))
      const b = basenameId(id)
      if (b && imageUncertaintyGlobal.byBase.has(b)) return Number(imageUncertaintyGlobal.byBase.get(b))
      return null
    }
    return Math.max(0, Math.min(1, sum / count))
  }
  let zoomSliderDrafts = {}
  let zoomSavingAxisIds = new Set()
  let zoomError = ''
  function clearZoomDraft(axisId) {
    const key = String(axisId || '')
    const next = { ...zoomSliderDrafts }
    delete next[key]
    zoomSliderDrafts = next
  }
  function setZoomDraft(axisId, value) {
    zoomSliderDrafts = { ...zoomSliderDrafts, [String(axisId || '')]: Math.max(0, Math.min(100, Number(value) || 0)) }
  }
  function axisEditorEntriesFor(imageId) {
    const ids = Array.from(new Set([
      ...[selectedX, selectedY].filter(Boolean),
      ...(Array.isArray(axisBuilderSessions) ? axisBuilderSessions.map((entry) => entry?.axisId).filter(Boolean) : []),
    ]))
    return ids
      .map((axisId) => {
        const builder = findBuilder(axisId)
        if (!builder) return null
        const currentScore = scoreForBuilderImage(builder, imageId)
        if (!Number.isFinite(currentScore)) return null
        return {
          axisId,
          axisName: builder?.axis?.name || axisName(axisId),
          currentScore,
          builder,
        }
      })
      .filter(Boolean)
  }
  $: zoomAxisEditors = zoomItemId ? axisEditorEntriesFor(zoomItemId) : []
  $: zoomItem = zoomItemId ? (itemsById.get(String(zoomItemId)) || renderItemsVisible.find((it) => it.id === zoomItemId) || null) : null
  async function applyAxisMove(axisId, imageId, target0to100, fallbackScore) {
    const current = findBuilder(axisId)
    if (!current || !imageId) return false
    const canonicalImageId = resolveBuilderImageId(current, imageId)
    const target = Math.max(0, Math.min(100, Number(target0to100) || 0))
    zoomError = ''
    zoomSavingAxisIds = new Set([...zoomSavingAxisIds, String(axisId)])
    try {
      const data = await postJson('/axis/move', {
        session: sessionName,
        axis_id: axisId,
        image_id: canonicalImageId,
        new_score_0_100: target,
      })
      const updated = axisSessionFromResponse(current, data, {
        preferredName: String(current?.axis?.name || current?.q || '').trim(),
      })
      if (!updated) throw new Error('Invalid axis move response')
      const nextHistory = Array.isArray(current?.moveHistory) ? [...current.moveHistory] : []
      nextHistory.unshift({
        imageId: canonicalImageId,
        fromScore0To100: Number(fallbackScore || 0),
        toScore0To100: Number(target || 0),
      })
      updated.moveHistory = nextHistory.slice(0, 6)
      axisBuildersStore.upsert(updated)
      if (updated?.axis?.id) dispatch('create', axisPayloadFromSession(updated))
      clearZoomDraft(axisId)
      return true
    } catch (err) {
      zoomError = `Axis update failed: ${String(err)}`
      return false
    } finally {
      const next = new Set(zoomSavingAxisIds)
      next.delete(String(axisId))
      zoomSavingAxisIds = next
    }
  }
  async function commitZoomAxisEdit(axisId, imageId, fallbackScore) {
    const target = Math.max(0, Math.min(100, Number(zoomSliderDrafts[String(axisId)] ?? fallbackScore) || 0))
    const ok = await applyAxisMove(axisId, imageId, target, fallbackScore)
    if (ok) emitLog('slider change', 'none')
  }
  function hashUnit01(seed) {
    const s = String(seed || '')
    let h = 2166136261 >>> 0
    for (let i = 0; i < s.length; i += 1) {
      h ^= s.charCodeAt(i)
      h = Math.imul(h, 16777619)
    }
    return (h >>> 0) / 4294967295
  }
  function subsampleHash(id) {
    return hashUnit01(`subsample|${String(imageSubsampleSeed || 0)}|${String(id || '')}`)
  }
  function jitteredAxisCoord(baseValue, itemId, axisId, dim) {
    const v = Number(baseValue)
    if (!Number.isFinite(v)) return Number.isFinite(baseValue) ? baseValue : 0.5
    if (!axisId || !isNonContinuousAxis(axisId)) return Math.max(0, Math.min(1, v))
    const ax = axesById.get(axisId)
    const labelsN = Array.isArray(ax?.labels) ? ax.labels.length : 0
    const ampFromBins = labelsN > 0 ? (0.38 / Math.max(4, labelsN)) : 0.024
    const amp = Math.max(0.006, Math.min(0.03, ampFromBins))
    const signed = (hashUnit01(`${axisId}|${itemId}|${dim}`) * 2) - 1
    return Math.max(0, Math.min(1, v + signed * amp))
  }
  function interp1(xs, ys, x) {
    if (!Array.isArray(xs) || !Array.isArray(ys) || xs.length === 0 || ys.length === 0) return 0
    if (xs.length === 1) return Number(ys[0]) || 0
    const xv = Number(x)
    if (!Number.isFinite(xv)) return 0
    if (xv <= xs[0]) return Number(ys[0]) || 0
    const last = xs.length - 1
    if (xv >= xs[last]) return Number(ys[last]) || 0
    let lo = 0
    let hi = last
    while (lo + 1 < hi) {
      const mid = (lo + hi) >> 1
      if (xs[mid] <= xv) lo = mid
      else hi = mid
    }
    const x0 = xs[lo]
    const x1 = xs[hi]
    const y0 = Number(ys[lo]) || 0
    const y1 = Number(ys[hi]) || 0
    const t = (xv - x0) / Math.max(1e-9, x1 - x0)
    return y0 + (y1 - y0) * t
  }
  $: sameAxisKdeLayout = (function buildSameAxisKdeLayout() {
    const axisId = (selectedX && selectedY && selectedX === selectedY) ? selectedX : null
    if (!axisId) return null
    const ax = axesById.get(axisId)
    if (!ax || !ax.coords) return null
    const pairs = []
    for (const it of (itemsFiltered || [])) {
      const xv = getCoord(it.id, axisId)
      if (typeof xv !== 'number' || !Number.isFinite(xv)) continue
      pairs.push({ id: it.id, x: Math.max(0, Math.min(1, xv)) })
    }
    if (pairs.length === 0) return null
    const xs = pairs.map((p) => p.x)
    const n = xs.length
    const mean = xs.reduce((s, v) => s + v, 0) / Math.max(1, n)
    let varSum = 0
    for (const v of xs) {
      const dv = v - mean
      varSum += dv * dv
    }
    const std = Math.sqrt(varSum / Math.max(1, n))
    const bandwidth = Math.max(0.015, Math.min(0.2, 1.06 * Math.max(std, 1e-3) * Math.pow(Math.max(2, n), -0.2)))
    const gridN = 128
    const gx = new Array(gridN)
    const gy = new Array(gridN)
    let maxD = 0
    const invH = 1 / Math.max(1e-6, bandwidth)
    for (let i = 0; i < gridN; i += 1) {
      const x = i / (gridN - 1)
      gx[i] = x
      let dens = 0
      for (let j = 0; j < n; j += 1) {
        const u = (x - xs[j]) * invH
        dens += Math.exp(-0.5 * u * u)
      }
      gy[i] = dens
      if (dens > maxD) maxD = dens
    }
    const out = new Map()
    const scale = maxD > 0 ? (0.92 / maxD) : 0
    for (const p of pairs) {
      const dens = interp1(gx, gy, p.x)
      const ymax = Math.max(0, Math.min(0.92, dens * scale))
      const u = hashUnit01(`${axisId}|${p.id}|kde-stack`)
      out.set(p.id, { x: p.x, y: u * ymax })
    }
    return out
  })()

  // Hover and zoom state
  let griddingActive = false
  let activeCenterId = null
  let gridRect = null // fixed rect while grid is active
  let griddedImsRect = null // last rect used to compute gridded images
  let gridMargin = 0.01
  // Zoom center for scaling points
  let cx = 0.5
  let cy = 0.5
  let hoverPointerNorm = { x: 0.5, y: 0.5 }
  let vf = viewFrac
  let minimapEl
  let focusRectEl
  let grabMode = false
  let grabDrag = null // { id, pointerId, x, y }
  let grabSaving = false

  function pointerClientToWorld(clientX, clientY) {
    const rect = minimapEl ? minimapEl.getBoundingClientRect() : null
    if (!rect || rect.width <= 0 || rect.height <= 0) return null
    const vx = (Number(clientX) - rect.left) / rect.width
    const vy = (Number(clientY) - rect.top) / rect.height
    const x = screenNormToWorldX(vx)
    const yTop = screenNormToWorldYTop(vy)
    return {
      x: clamp(x, 0, 1),
      y: clamp(1 - yTop, 0, 1),
    }
  }
  async function commitGrabDrag(drag) {
    if (!drag || !drag.id) return
    const imageId = String(drag.id)
    const ops = []
    const bx = selectedX ? findBuilder(selectedX) : null
    if (bx) {
      ops.push({
        axisId: selectedX,
        fallbackScore: scoreForBuilderImage(bx, imageId),
        target0to100: Number(drag.x || 0) * 100,
      })
    }
    if (selectedY && selectedY !== selectedX) {
      const by = findBuilder(selectedY)
      if (by) {
        ops.push({
          axisId: selectedY,
          fallbackScore: scoreForBuilderImage(by, imageId),
          target0to100: Number(drag.y || 0) * 100,
        })
      }
    }
    if (ops.length === 0) return
    grabSaving = true
    try {
      let moved = false
      for (const op of ops) {
        const ok = await applyAxisMove(op.axisId, imageId, op.target0to100, op.fallbackScore)
        moved = moved || ok
      }
      if (moved) emitLog('grabbing', 'none')
    } finally {
      grabSaving = false
    }
  }
  function stopGrabListeners() {
    window.removeEventListener('pointermove', onGrabPointerMove)
    window.removeEventListener('pointerup', onGrabPointerUp)
    window.removeEventListener('pointercancel', onGrabPointerUp)
  }
  function hitItemIdAt(px, py, rect) {
    let hitId = null
    for (let i = renderItemsVisible.length - 1; i >= 0; i--) {
      const it = renderItemsVisible[i]
      const inside = griddingActive && insideIds.has(it.id)
      const sz = renderThumbSizePx(it.id, inside, sizeInside, sizeOutside, subsampleDotPx, subsampleActive)
      const dx = Math.abs(px - it.x) * rect.width
      const dy = Math.abs(py - 1 + it.y) * rect.height
      if (dx <= sz / 2 && dy <= sz / 2) { hitId = it.id; break }
    }
    return hitId
  }
  function computeHoverWorldRect(vx, vy) {
    const centerX = screenNormToWorldX(vx)
    const centerY = screenNormToWorldYTop(vy)
    const viewW = vf / Math.max(1, zoomZ)
    const viewH = vf / Math.max(1, zoomZ)
    const x0 = Math.max(0, Math.min(1 - viewW, centerX - viewW / 2))
    const y0 = Math.max(0, Math.min(1 - viewH, centerY - viewH / 2))
    return {
      x0,
      y0,
      x1: x0 + viewW,
      y1: y0 + viewH,
      margin: hoverMargin,
    }
  }
  function worldRectToVisual(rect) {
    const left = worldXToScreenNorm(rect.x0)
    const right = worldXToScreenNorm(rect.x1)
    const top = worldYTopToScreenNorm(rect.y0)
    const bottom = worldYTopToScreenNorm(rect.y1)
    const x0 = Math.max(0, Math.min(1, Math.min(left, right)))
    const x1 = Math.max(0, Math.min(1, Math.max(left, right)))
    const y0 = Math.max(0, Math.min(1, Math.min(top, bottom)))
    const y1 = Math.max(0, Math.min(1, Math.max(top, bottom)))
    return {
      x0,
      y0,
      w: Math.max(0, x1 - x0),
      h: Math.max(0, y1 - y0),
    }
  }
  function syncHoverPreview(vx = hoverPointerNorm.x, vy = hoverPointerNorm.y) {
    hoverPointerNorm = {
      x: clamp(Number(vx || 0), 0, 1),
      y: clamp(Number(vy || 0), 0, 1),
    }
    if (!focusRectEl) return
    const rect = worldRectToVisual(computeHoverWorldRect(hoverPointerNorm.x, hoverPointerNorm.y))
    focusRectEl.style.left = `${rect.x0 * 100}%`
    focusRectEl.style.top = `${rect.y0 * 100}%`
    focusRectEl.style.width = `${rect.w * 100}%`
    focusRectEl.style.height = `${rect.h * 100}%`
  }
  function onGrabPointerMove(e) {
    if (!grabDrag) return
    if (grabDrag.pointerId !== undefined && e.pointerId !== undefined && e.pointerId !== grabDrag.pointerId) return
    const p = pointerClientToWorld(e.clientX, e.clientY)
    if (!p) return
    grabDrag = { ...grabDrag, x: p.x, y: p.y }
    ensureAnimationFrame()
    try { e.preventDefault() } catch (_) {}
  }
  async function onGrabPointerUp(e) {
    if (!grabDrag) return
    if (grabDrag.pointerId !== undefined && e.pointerId !== undefined && e.pointerId !== grabDrag.pointerId) return
    const p = pointerClientToWorld(e.clientX, e.clientY)
    const done = p ? { ...grabDrag, x: p.x, y: p.y } : { ...grabDrag }
    grabDrag = null
    stopGrabListeners()
    suppressUntil = performance.now() + 260
    await commitGrabDrag(done)
    try { e.preventDefault() } catch (_) {}
  }
  function onGrabStart(e) {
    if (!grabMode || zoomItemId || grabSaving) return
    const rect = e.currentTarget.getBoundingClientRect()
    const px = (e.clientX - rect.left) / rect.width
    const py = (e.clientY - rect.top) / rect.height
    const id = hitItemIdAt(px, py, rect)
    if (!id) return
    const p = pointerClientToWorld(e.clientX, e.clientY)
    if (!p) return
    grabDrag = { id, pointerId: e.pointerId, x: p.x, y: p.y }
    stopGrabListeners()
    window.addEventListener('pointermove', onGrabPointerMove)
    window.addEventListener('pointerup', onGrabPointerUp)
    window.addEventListener('pointercancel', onGrabPointerUp)
    suppressUntil = performance.now() + 120
    try { e.preventDefault() } catch (_) {}
    try { e.stopPropagation() } catch (_) {}
  }

  function onMove(e) {
    if (lassoEnabled || grabMode) return
    const rect = e.currentTarget.getBoundingClientRect()
    const vx = (e.clientX - rect.left) / rect.width
    const vy = (e.clientY - rect.top) / rect.height
    syncHoverPreview(vx, vy)
  }
  let zoomZ = 1.0 // visual zoom scale around (cx,cy); 1 = no zoom
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)) }
  // Buttons: resize the viewport (blue rect)
  function viewportZoomIn() { vf = clamp(vf * 0.88, 0.04, 0.9) }
  function viewportZoomOut() { vf = clamp(vf / 0.88, 0.04, 0.9) }
  let gridDismissRaf = 0
  function clearScheduledGridDismiss() {
    if (!gridDismissRaf) return
    cancelAnimationFrame(gridDismissRaf)
    gridDismissRaf = 0
  }
  function clearGridPacking({ closeZoomOverlay = false } = {}) {
    clearScheduledGridDismiss()
    griddingActive = false
    activeCenterId = null
    gridRect = null
    griddedImsRect = null
    if (closeZoomOverlay && zoomItemId) closeZoom()
    syncHoverPreview()
  }
  function currentGridPackingVisualRect() {
    if (!griddingActive || !gridRect) return null
    const base = gridBackdrop || {
      x0: worldXToScreenNorm(gridRect.x0),
      y0: worldYTopToScreenNorm(gridRect.y0),
      x1: worldXToScreenNorm(gridRect.x1),
      y1: worldYTopToScreenNorm(gridRect.y1),
    }
    return {
      x0: Math.min(Number(base.x0 ?? 0), Number(base.x1 ?? 0)),
      y0: Math.min(Number(base.y0 ?? 0), Number(base.y1 ?? 0)),
      x1: Math.max(Number(base.x0 ?? 0), Number(base.x1 ?? 0)),
      y1: Math.max(Number(base.y0 ?? 0), Number(base.y1 ?? 0)),
    }
  }
  function isClientPointInsideGridPacking(clientX, clientY) {
    if (!griddingActive || !gridRect || !minimapEl) return false
    const minimapRect = minimapEl.getBoundingClientRect()
    if (!minimapRect || minimapRect.width <= 0 || minimapRect.height <= 0) return false
    const visualRect = currentGridPackingVisualRect()
    if (!visualRect) return false
    const px = (Number(clientX) - minimapRect.left) / minimapRect.width
    const py = (Number(clientY) - minimapRect.top) / minimapRect.height
    return px >= visualRect.x0 && px <= visualRect.x1 && py >= visualRect.y0 && py <= visualRect.y1
  }
  function scheduleGridPackingDismiss(closeZoomOverlay = false) {
    clearScheduledGridDismiss()
    gridDismissRaf = requestAnimationFrame(() => {
      gridDismissRaf = 0
      if (!griddingActive) return
      clearGridPacking({ closeZoomOverlay })
    })
  }
  function onWindowClickCapture(e) {
    if (!griddingActive) return
    if (isClientPointInsideGridPacking(e.clientX, e.clientY)) return
    scheduleGridPackingDismiss(true)
  }
  function onWindowKeyDown(e) {
    if (!griddingActive) return
    if (e.key !== 'Escape') return
    try { e.preventDefault() } catch (_) {}
    clearGridPacking({ closeZoomOverlay: true })
  }
  // Wheel: zoom content around center
  function contentZoomIn() { zoomZ = clamp(zoomZ * 1.12, 1.0, 6.0) }
  function contentZoomOut() { zoomZ = clamp(zoomZ / 1.12, 1.0, 6.0) }
  function resetView() {
    cx = 0.5
    cy = 0.5
    zoomZ = 1.0
    vf = viewFrac
    clearGridPacking()
  }
  function applyRestoredViewState(raw) {
    const next = raw && typeof raw === 'object' ? raw : {}
    cx = clamp(Number(next.cx ?? 0.5), 0, 1)
    cy = clamp(Number(next.cy ?? 0.5), 0, 1)
    zoomZ = clamp(Number(next.zoomZ ?? 1.0), 1.0, 6.0)
    vf = clamp(Number(next.vf ?? viewFrac), 0.04, 0.9)
    showDensity = !!next.showDensity
    showUncertainty = false
    imageMax = Math.max(0, Math.floor(Number(next.imageMax ?? imageMax) || 0))
    cellPx = clamp(Math.round(Number(next.cellPx ?? cellPx) || 16), MIN_CELL_PX, MAX_CELL_PX)
    selectedColorAxisId = String(next.selectedColorAxisId || '').trim() || null
    clearLassoSelection()
    clearGridPacking()
    zoomItemId = null
    zoomSliderDrafts = {}
    zoomError = ''
  }
  let appliedRestoreNonce = null
  $: {
    const state = restoreViewState
    const nonce = state && typeof state === 'object' ? Number(state.nonce || 0) : 0
    if (state && nonce && nonce !== appliedRestoreNonce) {
      appliedRestoreNonce = nonce
      applyRestoredViewState(state)
    }
  }
  function onWheel(e) {
    if (e.shiftKey) {
      try { e.preventDefault() } catch (_) {}
      if (e.deltaY < 0) viewportZoomIn()
      else viewportZoomOut()
      return
    }
    // Zoom around the cursor so the pointed spot stays fixed
    const rect = e.currentTarget.getBoundingClientRect()
    const vx = (e.clientX - rect.left) / Math.max(1, rect.width)
    const vy = (e.clientY - rect.top) / Math.max(1, rect.height)
    const zx = fromVisX(vx)
    const zy = fromVisY(1 - vy) // convert screen-down to internal up
    const factor = 1.12
    const nextZ = clamp(e.deltaY < 0 ? (zoomZ * factor) : (zoomZ / factor), 1.0, 6.0)
    if (nextZ === zoomZ) return
    // Convert current cursor zoom-space position (zx,zy) to world (pre-zoom) coords
    const px = cx + (zx - cx) / Math.max(1e-6, zoomZ)
    const py = cy + (zy - cy) / Math.max(1e-6, zoomZ)
    if (Math.abs(1 - nextZ) > 1e-6) {
      const denom = (1 - nextZ)
      cx = clamp((zx - nextZ * px) / denom, 0, 1)
      cy = clamp((zy - nextZ * py) / denom, 0, 1)
    }
    zoomZ = nextZ
  }

  // KDE density overlay (contours)
  let showDensity = false
  let densityCanvas
  let pointsCanvas
  let pointRadius = 0 // 0 = images only; >0 draws points with given radius (px)
  let isoCount = 5 // number of contour levels
  function gaussianKernel(sigma) {
    const r = Math.max(1, Math.round(sigma * 2.5))
    const k = new Float32Array(2*r + 1)
    const s2 = sigma * sigma * 2
    let sum = 0
    for (let i = -r; i <= r; i++) { const v = Math.exp(-(i*i)/s2); k[i + r] = v; sum += v }
    for (let i = 0; i < k.length; i++) k[i] /= sum
    return { k, r }
  }
  function convolve1D(arr, w, h, kernel) {
    const out = new Float32Array(arr.length)
    const { k, r } = kernel
    // Horizontal
    for (let y=0; y<h; y++) {
      for (let x=0; x<w; x++) {
        let acc = 0
        for (let t=-r; t<=r; t++) {
          const xx = Math.max(0, Math.min(w-1, x + t))
          acc += arr[y*w + xx] * k[t+r]
        }
        out[y*w + x] = acc
      }
    }
    // Vertical into arr
    const out2 = new Float32Array(arr.length)
    for (let y=0; y<h; y++) {
      for (let x=0; x<w; x++) {
        let acc = 0
        for (let t=-r; t<=r; t++) {
          const yy = Math.max(0, Math.min(h-1, y + t))
          acc += out[yy*w + x] * k[t+r]
        }
        out2[y*w + x] = acc
      }
    }
    return out2
  }
  function buildDensityGrid(points, wpx, hpx) {
    const gw = Math.min(180, Math.max(60, Math.round(wpx/6)))
    const gh = Math.min(180, Math.max(60, Math.round(hpx/6)))
    const grid = new Float32Array(gw*gh)
    const clamp01f = (v) => Math.max(0, Math.min(1, v))
    for (const p of points) {
      const gx = clamp01f(p.x) * (gw - 1)
      const gy = clamp01f(1 - p.y) * (gh - 1)
      const x0 = Math.floor(gx), y0 = Math.floor(gy)
      const dx = gx - x0, dy = gy - y0
      const x1 = Math.min(gw - 1, x0 + 1)
      const y1 = Math.min(gh - 1, y0 + 1)
      const w00 = (1 - dx) * (1 - dy)
      const w10 = dx * (1 - dy)
      const w01 = (1 - dx) * dy
      const w11 = dx * dy
      grid[y0*gw + x0] += w00
      grid[y0*gw + x1] += w10
      grid[y1*gw + x0] += w01
      grid[y1*gw + x1] += w11
    }
    // Blur with Gaussian kernel ~ 2% of min dimension in grid cells
    const sigma = Math.max(1.2, Math.min(15.5, Math.min(gw, gh) * 0.03))
    const ker = gaussianKernel(sigma)
    const blurred = convolve1D(grid, gw, gh, ker)
    // Normalize
    let maxv = 1e-6
    for (let i=0;i<blurred.length;i++) if (blurred[i] > maxv) maxv = blurred[i]
    for (let i=0;i<blurred.length;i++) blurred[i] /= maxv
    return { data: blurred, gw, gh }
  }
  function drawContours(ctx, grid, gw, gh, levels, wpx, hpx) {
    // Marching squares
    function interp(p1, p2, v1, v2, t) { const a = (t - v1) / ((v2 - v1) || 1e-6); return { x: p1.x + a * (p2.x - p1.x), y: p1.y + a * (p2.y - p1.y) } }
    ctx.save()
    const dprLocal = (typeof devicePixelRatio !== 'undefined' ? devicePixelRatio : 1)
    ctx.lineWidth = Math.max(1, Math.floor(dprLocal))
    // First paint a subtle blue fill for regions above each threshold
    const cellW = wpx / Math.max(1, (gw - 1))
    const cellH = hpx / Math.max(1, (gh - 1))
    for (const t of levels) {
      ctx.save()
      ctx.fillStyle = 'rgba(59,130,246,0.10)'
      for (let y=0; y<gh-1; y++) {
        for (let x=0; x<gw-1; x++) {
          const i00 = y*gw + x, i10 = y*gw + (x+1), i11 = (y+1)*gw + (x+1), i01 = (y+1)*gw + x
          const avg = (grid[i00] + grid[i10] + grid[i11] + grid[i01]) * 0.25
          if (avg >= t) {
            ctx.fillRect(x * cellW, y * cellH, cellW, cellH)
          }
        }
      }
      ctx.restore()
    }
    // Then draw iso-lines on top
    ctx.strokeStyle = 'rgba(59,130,246,0.85)'
    for (const t of levels) {
      for (let y=0; y<gh-1; y++) {
        for (let x=0; x<gw-1; x++) {
          const i00 = y*gw + x, i10 = y*gw + (x+1), i11 = (y+1)*gw + (x+1), i01 = (y+1)*gw + x
          const v00 = grid[i00] - t, v10 = grid[i10] - t, v11 = grid[i11] - t, v01 = grid[i01] - t
          const idx = (v00>0?8:0) | (v10>0?4:0) | (v11>0?2:0) | (v01>0?1:0)
          if (idx === 0 || idx === 15) continue
          const p00 = { x: x/(gw-1)*wpx, y: y/(gh-1)*hpx }
          const p10 = { x: (x+1)/(gw-1)*wpx, y: y/(gh-1)*hpx }
          const p11 = { x: (x+1)/(gw-1)*wpx, y: (y+1)/(gh-1)*hpx }
          const p01 = { x: x/(gw-1)*wpx, y: (y+1)/(gh-1)*hpx }
          // Edge intersections
          const e = {}
          e.top = interp(p00, p10, v00, v10, 0)
          e.right = interp(p10, p11, v10, v11, 0)
          e.bottom = interp(p01, p11, v01, v11, 0)
          e.left = interp(p00, p01, v00, v01, 0)
          // Cases mapping (with simple split for ambiguous 5/10)
          const cases = {
            1: ['left','bottom'], 2: ['bottom','right'], 3: ['left','right'], 4: ['top','right'], 5: ['top','left','bottom','right'], 6: ['top','bottom'], 7: ['left','top'],
            8: ['left','top'], 9: ['top','bottom'], 10: ['top','right','left','bottom'], 11: ['top','right'], 12: ['left','right'], 13: ['bottom','right'], 14: ['left','bottom']
          }
          const seg = cases[idx]
          if (!seg) continue
          ctx.beginPath()
          if (seg.length === 2) {
            const a = e[seg[0]], b = e[seg[1]]
            ctx.moveTo(a.x, a.y)
            ctx.lineTo(b.x, b.y)
            ctx.stroke()
          } else if (seg.length === 4) {
            // ambiguous cell: draw two segments
            const a1 = e[seg[0]], b1 = e[seg[1]]
            const a2 = e[seg[2]], b2 = e[seg[3]]
            ctx.moveTo(a1.x, a1.y); ctx.lineTo(b1.x, b1.y); ctx.stroke()
            ctx.beginPath(); ctx.moveTo(a2.x, a2.y); ctx.lineTo(b2.x, b2.y); ctx.stroke()
          }
        }
      }
    }
    ctx.restore()
  }
  $: densityScatterPoints = Array.isArray(itemsFiltered)
    ? (() => {
        touch(scatterLayoutDeps)
        return itemsFiltered
          .map((it) => visibleScatterPositionForItem(it))
          .filter(Boolean)
          .map((pt) => ({ x: pt.x, y: pt.y }))
      })()
    : []
  $: if (showDensity && densityCanvas && width && height) {
    try {
      const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) ? window.devicePixelRatio : 1
      const wpx = Math.max(1, Math.floor(width))
      const hpx = Math.max(1, Math.floor(height))
      densityCanvas.width = Math.floor(wpx * dpr)
      densityCanvas.height = Math.floor(hpx * dpr)
      densityCanvas.style.width = wpx + 'px'
      densityCanvas.style.height = hpx + 'px'
      const ctx = densityCanvas.getContext('2d')
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, wpx, hpx)
      const pts = densityScatterPoints
      if (pts.length > 0) {
        const { data, gw, gh } = buildDensityGrid(pts, wpx, hpx)
        const n = Math.max(1, Math.min(20, Math.floor(isoCount || 1)))
        const levels = Array.from({ length: n }, (_, i) => 0.15 + (0.7 * (i / Math.max(1, n - 1))))
        drawContours(ctx, data, gw, gh, levels, wpx, hpx)
      }
    } catch (_) { /* ignore drawing errors */ }
  }
  $: if (!showDensity && densityCanvas && width && height) {
    try {
      const ctx = densityCanvas.getContext('2d')
      if (ctx) ctx.clearRect(0, 0, densityCanvas.width, densityCanvas.height)
    } catch (_) {}
  }
  function metadataColorForId(imageId) {
    const key = String(imageId || '').trim()
    if (!key) return null
    return metadataColorLookup.get(key) || null
  }
  function pointColorForId(imageId, fallback) {
    const meta = metadataColorForId(imageId)
    return meta?.pointColor || fallback
  }
  function pointStrongColorForId(imageId, fallback) {
    const meta = metadataColorForId(imageId)
    return meta?.pointColorStrong || fallback
  }

  // Points overlay drawing
  $: {
    touch(scatterLayoutDeps)
    const drawAllPoints = pointRadius > 0
    const drawSubsampleDots = !drawAllPoints && subsampleActive && !griddingActive
    if (pointsCanvas && width && height && (drawAllPoints || drawSubsampleDots)) {
    try {
      const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) ? window.devicePixelRatio : 1
      const wpx = Math.max(1, Math.floor(width))
      const hpx = Math.max(1, Math.floor(height))
      pointsCanvas.width = Math.floor(wpx * dpr)
      pointsCanvas.height = Math.floor(hpx * dpr)
      pointsCanvas.style.width = wpx + 'px'
      pointsCanvas.style.height = hpx + 'px'
      const ctx = pointsCanvas.getContext('2d')
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, wpx, hpx)
      if (drawAllPoints) {
        const r = Math.max(1, Math.round(Math.max(0, pointRadius)))
        for (const it of renderItemsVisible) {
          const x = it.x * wpx
          const y = (1 - it.y) * hpx
          if (isSliceFilteredOut(it.id)) {
            ctx.fillStyle = 'rgba(148,163,184,0.35)'
            ctx.beginPath()
            ctx.arc(x, y, r, 0, Math.PI * 2)
            ctx.fill()
            continue
          }
          // Color by label: neg=red, pos=green, unlabeled=black
          const lv = labelOf(it.id)
          const fallback = (lv === 'neg') ? 'rgba(220,38,38,0.95)'
                         : (lv === 'pos') ? 'rgba(22,163,74,0.95)'
                         : 'rgba(17,24,39,0.95)'
          const col = pointStrongColorForId(it.id, fallback)
          ctx.fillStyle = col
          ctx.beginPath()
          ctx.arc(x, y, r, 0, Math.PI * 2)
          ctx.fill()
        }
      } else if (drawSubsampleDots) {
        const dotR = Math.max(1, Math.floor(Math.max(2, subsampleDotPx * 0.5)))
        for (const it of (itemsFiltered || [])) {
          if (sampledIds.has(it.id)) continue
          const p = posOriginal(it)
          const zx = cx + (p.x - cx) * zoomZ
          const zy = cy + (p.y - cy) * zoomZ
          const sx = toVisX(zx)
          const sy = toVisY(zy)
          if (sx < 0 || sx > 1 || sy < 0 || sy > 1) continue
          const x = sx * wpx
          const y = (1 - sy) * hpx
          ctx.fillStyle = isSliceFilteredOut(it.id)
            ? 'rgba(148,163,184,0.30)'
            : pointColorForId(it.id, 'rgba(143,151,170,0.74)')
          ctx.beginPath()
          ctx.arc(x, y, dotR, 0, Math.PI * 2)
          ctx.fill()
        }
      }
    } catch (_) { /* ignore */ }
    } else if (pointsCanvas) {
      try { const ctx = pointsCanvas.getContext('2d'); ctx && ctx.clearRect(0, 0, pointsCanvas.width, pointsCanvas.height) } catch (_) {}
    }
  }

  // Viewport rect
  $: viewW = vf / Math.max(1, zoomZ)
  $: viewH = vf / Math.max(1, zoomZ)
  $: hoverMargin = Math.min(0.02, (vf * 0.15) / Math.max(1, zoomZ))
  $: {
    viewW
    viewH
    hoverMargin
    zoomZ
    cx
    cy
    width
    height
    squareWNorm
    squareHNorm
    squareXOffsetNorm
    squareYOffsetNorm
    focusRectEl
    syncHoverPreview()
  }

  // Positions selection helpers
  function posOriginal(it) {
    if (!showUncertainty && selectedX && selectedY && selectedX === selectedY && sameAxisKdeLayout) {
      const p = sameAxisKdeLayout.get(it.id)
      if (p) return p
    }
    // Derive x from selected X axis (fallback to item.x), and y from selected Y axis (fallback to item.y)
    const ox = getCoord(it.id, selectedX)
    const oy = getCoord(it.id, selectedY)
    const xRaw = (ox !== undefined) ? ox : Number(it.x ?? 0)
    const yRaw = (oy !== undefined) ? oy : Number(it.y ?? 0)
    const x = (ox !== undefined) ? jitteredAxisCoord(xRaw, it.id, selectedX, 'x') : Math.max(0, Math.min(1, Number(xRaw || 0)))
    let y = (oy !== undefined) ? jitteredAxisCoord(yRaw, it.id, selectedY, 'y') : Math.max(0, Math.min(1, Number(yRaw || 0)))
    if (showUncertainty) {
      const u = uncertaintyNormForImage(it.id)
      if (Number.isFinite(u)) y = Math.max(0, Math.min(1, Number(u)))
    }
    return { x, y }
  }
  function targetWorldPositionForItem(it) {
    const base = posOriginal(it)
    const rect = (griddingActive && gridRect) ? gridRect : null
    const rx0 = rect?.x0 ?? 0
    const ry0 = rect?.y0 ?? 0
    const rx1 = rect?.x1 ?? 1
    const ry1 = rect?.y1 ?? 1
    const rmg = rect?.margin ?? 0
    const inside = !!(
      griddingActive &&
      rect &&
      base.x >= (rx0 - rmg) &&
      base.x <= (rx1 + rmg) &&
      (1 - base.y) >= (ry0 - rmg) &&
      (1 - base.y) <= (ry1 + rmg)
    )
    const packed = inside ? localPacked.get(it.id) : null
    return packed || base
  }
  function screenPositionForItem(it) {
    const track = tracks?.get(it.id)
    const draggingThis = !!(grabDrag && String(grabDrag.id) === String(it?.id))
    const world = track
      ? {
          x: draggingThis ? Number(grabDrag.x) : lerp(track.from.x, track.to.x, tNorm),
          y: draggingThis ? Number(grabDrag.y) : lerp(track.from.y, track.to.y, tNorm),
        }
      : targetWorldPositionForItem(it)
    const zx = cx + (world.x - cx) * zoomZ
    const zy = cy + (world.y - cy) * zoomZ
    return {
      id: it.id,
      x: toVisX(zx),
      y: toVisY(zy),
    }
  }
  // Normalized items for the lasso overlay – use the actually rendered positions
  // so selection matches the current (possibly customized) axes projection and packing,
  // even for items that are currently represented as dots or omitted from thumbnail rendering.
  $: lassoItems = Array.isArray(itemsFiltered)
    ? itemsFiltered
      .map((it) => screenPositionForItem(it))
      .filter((it) => Number.isFinite(it.x) && Number.isFinite(it.y) && it.x >= 0 && it.x <= 1 && it.y >= 0 && it.y <= 1)
      .map((it) => ({ id: it.id, x: it.x, y: 1 - it.y }))
    : []

  // Animation tracks
  let lastPosMap = new Map() // id -> {x,y}
  let prevTargets = new Map() // id -> {x,y}
  let tracks = new Map() // id -> {id,url,from,to}
  function easeInOutCubic(t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2 }
  function lerp(a, b, t) { return a + (b - a) * t }
  function posEqual(a, b, eps = 1e-6) {
    if (!a || !b) return false
    return Math.abs(a.x - b.x) <= eps && Math.abs(a.y - b.y) <= eps
  }
  let startTs = performance.now()
  let nowTs = startTs
  let rafId = 0
  let resizeDrag = null
  function transitionProgress(ts = nowTs) {
    const elapsedMs = Math.max(0, Number(ts || 0) - Number(startTs || 0))
    const ratio = Math.min(1, duration > 0 ? elapsedMs / duration : 1)
    return easeInOutCubic(ratio)
  }
  function updateLastPositions(progress = transitionProgress()) {
    if (!(tracks && tracks.size > 0)) return
    for (const tr of tracks.values()) {
      const x = lerp(tr.from.x, tr.to.x, progress)
      const y = lerp(tr.from.y, tr.to.y, progress)
      lastPosMap.set(tr.id, { x, y })
    }
  }
  function ensureAnimationFrame() {
    if (rafId) return
    rafId = requestAnimationFrame(tickAnim)
  }
  function tickAnim() {
    rafId = 0
    const ts = performance.now()
    nowTs = ts
    const progress = transitionProgress(ts)
    updateLastPositions(progress)
    if (grabDrag || progress < 0.999) {
      ensureAnimationFrame()
      return
    }
    updateLastPositions(1)
  }
  onMount(() => {
    syncHoverPreview()
    window.addEventListener('click', onWindowClickCapture, true)
    window.addEventListener('keydown', onWindowKeyDown)
  })
  function clearResizeDrag() {
    resizeDrag = null
    window.removeEventListener('pointermove', onResizeDragMove)
    window.removeEventListener('pointerup', onResizeDragUp)
  }
  function onResizeHandleDown(e) {
    resizeDrag = {
      lastX: Number(e.clientX || 0),
      lastY: Number(e.clientY || 0),
    }
    window.addEventListener('pointermove', onResizeDragMove)
    window.addEventListener('pointerup', onResizeDragUp)
    try { e.preventDefault() } catch (_) {}
  }
  function onResizeDragMove(e) {
    if (!resizeDrag) return
    const x = Number(e.clientX || 0)
    const y = Number(e.clientY || 0)
    const dx = x - resizeDrag.lastX
    const dy = y - resizeDrag.lastY
    resizeDrag.lastX = x
    resizeDrag.lastY = y
    const deltaPx = Math.max(dx, dy)
    if (!Number.isFinite(deltaPx) || Math.abs(deltaPx) < 0.001) return
    dispatch('resizeMinimap', { deltaPx })
  }
  function onResizeDragUp() {
    clearResizeDrag()
  }
  onDestroy(() => {
    cancelAnimationFrame(rafId)
    clearScheduledGridDismiss()
    clearResizeDrag()
    stopGrabListeners()
    window.removeEventListener('click', onWindowClickCapture, true)
    window.removeEventListener('keydown', onWindowKeyDown)
  })

  // Local packing for inside grid placement
  function computeLocalPacked(itemsArr, rect, sizePx, spacingScale, wPx, hPx) {
    if (!rect) return new Map()
    const rx0 = rect.x0, ry0 = rect.y0, rx1 = rect.x1, ry1 = rect.y1
    const margin = rect.margin || 0
    const stepPx = Math.max(1, sizePx * spacingScale)
    const displayMarginClamped = Math.max(0, Math.min(0.49, Number(displayMargin || 0)))
    const visibleSquarePx = Math.max(1, squareSidePx * Math.max(1e-6, 1 - (2 * displayMarginClamped)))
    // Keep packed-image gaps stable on screen as the view zoom changes.
    const stepNorm = Math.max(1e-6, stepPx / (visibleSquarePx * Math.max(1, zoomZ)))
    const sx = stepNorm
    const sy = stepNorm
    const inside = itemsArr.filter((it) => {
      const p = posOriginal(it)
      return p.x >= (rx0 - margin) && p.x <= (rx1 + margin) && (1 - p.y) >= (ry0 - margin) && (1 - p.y) <= (ry1 + margin)
    })
    if (!inside.length) return new Map()
    const cx = (rx0 + rx1) / 2, cy = (ry0 + ry1) / 2
    let center = inside[0], best = Infinity
    for (const it of inside) {
      const p = posOriginal(it)
      const dx = p.x - cx
      const dy = (1 - p.y) - cy
      const d2 = dx*dx + dy*dy
      if (d2 < best) { best = d2; center = it }
    }
    const base = posOriginal(center)
    const baseTopY = 1 - base.y
    function cellToXY(ix, iy) {
      const nextX = base.x + ix * sx
      const nextTopY = baseTopY + iy * sy
      return {
        x: nextX,
        y: 1 - nextTopY,
      }
    }
    const occ = new Set(); const key = (ix,iy)=> ix+','+iy
    occ.add(key(0,0))
    const ordered = [...inside].sort((a,b)=>{
      const pa = posOriginal(a), pb = posOriginal(b)
      if (a.id===center.id) return -1; if (b.id===center.id) return 1
      const da=(pa.x-base.x)**2+((1-pa.y)-baseTopY)**2
      const db=(pb.x-base.x)**2+((1-pb.y)-baseTopY)**2
      return da-db
    })
    const out = new Map(); out.set(center.id, { x: base.x, y: base.y })
    for (const it of ordered) {
      if (it.id === center.id) continue
      const p = posOriginal(it)
      let ix = Math.round((p.x - base.x) / sx)
      let iy = Math.round(((1 - p.y) - baseTopY) / sy)
      let found = null
      const maxR = inside.length + 4
      for (let r=0; r<=maxR && !found; r++) {
        for (let dx=-r; dx<=r; dx++) {
          for (let dy=-r; dy<=r; dy++) {
            if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue
            const tx = ix + dx, ty = iy + dy
            if (occ.has(key(tx,ty))) continue
            found = { ix: tx, iy: ty }
            break
          }
          if (found) break
        }
      }
      if (!found) { found = { ix, iy } }
      occ.add(key(found.ix, found.iy)); out.set(it.id, cellToXY(found.ix, found.iy))
    }
    return out
  }

  // Derived layout
  $: imSize = Math.max(minImagePx, Math.floor(cellPx ))
  $: vfRatio = 0.35 / Math.max(0.08, Math.min(1.0, vf))
  $: insideScale = Math.max(1.5, 1.7 + 1 * (vfRatio - 1))
  $: sizeInside = Math.max(minImagePx, Math.floor(imSize * insideScale))
  $: sizeOutside = Math.max(8, Math.floor(imSize * 0.8))
  $: spacingScale = 1.04
  $: subsampleDotPx = 5
  $: normalizedImageMax = Math.max(0, Math.floor(Number(imageMax) || 0))
  let lastViewStateSnapshot = ''
  $: {
    const snapshot = {
      cx,
      cy,
      zoomZ,
      vf,
      showDensity,
      showUncertainty: false,
      imageMax: normalizedImageMax,
      cellPx,
      selectedColorAxisId: selectedColorAxisId || '',
    }
    const nextJson = JSON.stringify(snapshot)
    if (nextJson !== lastViewStateSnapshot) {
      lastViewStateSnapshot = nextJson
      dispatch('viewStateChange', snapshot)
    }
  }
  $: sampledIds = (() => {
    if (normalizedImageMax <= 0) return null
    const base = Array.isArray(itemsFiltered) ? itemsFiltered : []
    if (base.length <= normalizedImageMax) return null
    const scored = base
      .map((it) => ({ id: it.id, h: subsampleHash(it.id) }))
      .sort((a, b) => (a.h - b.h) || String(a.id).localeCompare(String(b.id)))
    return new Set(scored.slice(0, normalizedImageMax).map((row) => row.id))
  })()
  $: subsampleActive = sampledIds instanceof Set
  function showsImageThumb(id, inside = false) {
    if (!subsampleActive) return true
    // In gridding mode, always load full thumbnails for items inside the focus grid.
    if (griddingActive && inside) return true
    return sampledIds.has(id)
  }
  function renderThumbSizePx(id, inside, insidePx = sizeInside, outsidePx = sizeOutside, dotPx = subsampleDotPx, subsampled = subsampleActive) {
    const base = inside ? insidePx : outsidePx
    if (!subsampled) return Math.max(6, Math.floor(base))
    if (!showsImageThumb(id, inside)) return dotPx
    if (inside) return Math.max(8, Math.floor(base))
    return Math.max(8, Math.floor(base * 0.72))
  }

  $: renderSourceItems = (() => {
    const base = Array.isArray(itemsFiltered) ? itemsFiltered : []
    if (!subsampleActive) return base
    if (griddingActive && gridRect) {
      return base.filter((it) => insideIds.has(it.id) || sampledIds.has(it.id))
    }
    return base.filter((it) => sampledIds.has(it.id))
  })()

  // Precompute inside ids set (using original positions and fixed gridRect captured on click)
  $: insideIds = (() => {
    touch(scatterLayoutDeps)
    if (!griddingActive || !gridRect) return new Set()
    return new Set(itemsFiltered.filter((it) => {
      const p = posOriginal(it)
      const margin = Number(gridRect.margin || 0)
      return p.x >= (gridRect.x0 - margin) && p.x <= (gridRect.x1 + margin) && (1 - p.y) >= (gridRect.y0 - margin) && (1 - p.y) <= (gridRect.y1 + margin)
    }).map((it) => it.id))
  })()

  $: localPacked = (() => {
    touch(scatterLayoutDeps)
    return computeLocalPacked(itemsFiltered, (griddingActive && gridRect) ? gridRect : null, sizeInside, spacingScale, width, height)
  })()

  // Compute animation targets and renderItems
  $: {
    touch(scatterLayoutDeps)
    const nextTargets = new Map(); const m = new Map(); let anyChange = false
    for (const it of renderSourceItems) {
      const p = posOriginal(it)
      const rect = (griddingActive && gridRect) ? gridRect : null
      const rx0 = rect?.x0 ?? 0
      const ry0 = rect?.y0 ?? 0
      const rx1 = rect?.x1 ?? 1
      const ry1 = rect?.y1 ?? 1
      const rmg = rect?.margin ?? 0
      const inside = !!(griddingActive && rect && p.x >= (rx0 - rmg) && p.x <= (rx1 + rmg) && (1 - p.y) >= (ry0 - rmg) && (1 - p.y) <= (ry1 + rmg))
      const lp = localPacked.get(it.id)
      const to = (inside && lp) ? lp : p
      const prevTarget = prevTargets.get(it.id)
      if (prevTarget && !posEqual(prevTarget, to)) anyChange = true
      const from = lastPosMap.get(it.id) || p
      m.set(it.id, { id: it.id, url: it.url, thumbUrl: it.thumbUrl, fullUrl: it.fullUrl, from, to })
      nextTargets.set(it.id, to)
    }
    tracks = m
    if (anyChange) {
      startTs = performance.now()
      nowTs = startTs
      ensureAnimationFrame()
    } else {
      updateLastPositions(1)
    }
    prevTargets = nextTargets
  }

  $: elapsed = Math.max(0, nowTs - startTs)
  $: raw = Math.min(1, duration > 0 ? elapsed / duration : 1)
  $: tNorm = easeInOutCubic(raw)
  $: renderItems = Array.from(tracks.values()).map((tr) => {
    const draggingThis = !!(grabDrag && String(grabDrag.id) === String(tr.id))
    const wx = draggingThis ? Number(grabDrag.x) : lerp(tr.from.x, tr.to.x, tNorm)
    const wy = draggingThis ? Number(grabDrag.y) : lerp(tr.from.y, tr.to.y, tNorm)
    const zx = cx + (wx - cx) * zoomZ
    const zy = cy + (wy - cy) * zoomZ
    const sx = toVisX(zx)
    const sy = toVisY(zy)
    return { id: tr.id, url: tr.url, thumbUrl: tr.thumbUrl, fullUrl: tr.fullUrl, x: sx, y: sy }
  })
  $: renderItemsVisible = Array.isArray(renderItems) ? renderItems.filter((it) => it.x >= 0 && it.x <= 1 && it.y >= 0 && it.y <= 1) : []
  function clamp01(v) { return Math.max(0, Math.min(1, v)) }
  // Grey rectangle based on visible gridded images (static while grid is active)
  $: gridBackdrop = (function computeBackdrop(active, list, insideSet, sizePx, wPx, hPx) {
    if (!active || !Array.isArray(list) || list.length === 0) return null
    const halfWn = (sizePx / Math.max(1, wPx)) / 2
    const halfHn = (sizePx / Math.max(1, hPx)) / 2
    let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity
    let count = 0
    for (const it of list) {
      if (!insideSet.has(it.id)) continue
      count++
      if (it.x < minx) minx = it.x
      if (1-it.y < miny) miny = 1-it.y
      if (it.x > maxx) maxx = it.x
      if (1-it.y > maxy) maxy = 1-it.y
    }
    if (count === 0 || !isFinite(minx)) return null
    const x0b = clamp01(minx - halfWn)
    const y0b = clamp01(miny - halfHn)
    const x1b = clamp01(maxx + halfWn)
    const y1b = clamp01(maxy + halfHn)
    return { x0: x0b, y0: y0b, w: Math.max(0, x1b - x0b), h: Math.max(0, y1b - y0b), x1: x1b, y1: y1b }
  })(griddingActive, renderItemsVisible, insideIds, Math.floor(sizeInside), width, height)

  let suppressUntil = 0
  function pickOrToggle(e) {
    if (zoomItemId) { return } // When overlay is open, ignore background clicks
    if (lassoEnabled) { return }
    if (grabMode || grabSaving || grabDrag) { return }
    if (performance.now() < suppressUntil) { return }
    const rect = e.currentTarget.getBoundingClientRect()
    const px = (e.clientX - rect.left) / rect.width
    const py = (e.clientY - rect.top) / rect.height
    
    // Find nearest visible image under cursor (in visual coords)
    const hitId = hitItemIdAt(px, py, rect)
    // If grid is active, handle inside/outside clicks
    if (griddingActive && gridRect) {
      const vx = px
      const vy = py
      // Compare against gridded images rect for interaction; fallback to captured rect if missing
      const rx0 = gridBackdrop ? gridBackdrop.x0 : worldXToScreenNorm(gridRect.x0)
      const ry0 = gridBackdrop ? gridBackdrop.y0 : worldYTopToScreenNorm(gridRect.y0)
      const rx1 = gridBackdrop ? gridBackdrop.x1 : worldXToScreenNorm(gridRect.x1)
      const ry1 = gridBackdrop ? gridBackdrop.y1 : worldYTopToScreenNorm(gridRect.y1)
      const inside = (vx >= rx0 && vx <= rx1 && vy >= ry0 && vy <= ry1)
      if (!inside) {
        // Click outside grid -> close grid
        clearGridPacking()
        try { console.log('[minimap] grid closed') } catch(_) {}
        return
      }
      // Inside grid: if clicking a gridded image, open zoom overlay
      if (hitId && insideIds && insideIds.has(hitId)) {
        zoomItemId = hitId
        try { console.log('[minimap] zoom image', hitId) } catch(_) {}
        return
      }
      // Otherwise do nothing (keep grid)
      return
    }
    // Grid not active: enable grid on current viewport rectangle (capture current blue rect)
    syncHoverPreview(px, py)
    const clickedRect = computeHoverWorldRect(px, py)
    griddingActive = true
    activeCenterId = null
    gridRect = clickedRect
    emitLog('local gridding', 'none')
    try {
      console.log('[minimap] grid enable gridRect (norm)', clickedRect)
      // compute and log grey bounds based on current visible items
      const sz = Math.floor(sizeInside)
      const halfWn = (sz / Math.max(1, width)) / 2
      const halfHn = (sz / Math.max(1, height)) / 2
      let minx=Infinity, miny=Infinity, maxx=-Infinity, maxy=-Infinity, count=0
      for (const it of renderItemsVisible) { if (insideIds.has(it.id)) { count++; if (it.x<minx) minx=it.x; if (it.y<miny) miny=it.y; if (it.x>maxx) maxx=it.x; if (it.y>maxy) maxy=it.y } }
      if (count>0 && isFinite(minx)) {
        const x0b = clamp01(minx - halfWn), y0b = clamp01(miny - halfHn)
        const x1b = clamp01(maxx + halfWn), y1b = clamp01(maxy + halfHn)
        console.log('[minimap] grey visual %', { x0: +(x0b*100).toFixed(1), y0: +(y0b*100).toFixed(1), x1: +(x1b*100).toFixed(1), y1: +(y1b*100).toFixed(1) })
      }
      console.log('[minimap] insideIds count', insideIds.size, 'sample', Array.from(insideIds).slice(0,10))
    } catch(_) {}
  }

  
  let zoomItemId = null
  let consumedFocusNonce = null
  $: {
    const req = focusImageRequest
    const imageId = req && typeof req === 'object' ? normalizeImageId(req.imageId) : ''
    const nonce = req && typeof req === 'object' ? Number(req.nonce || 0) : 0
    if (imageId && Number.isFinite(nonce) && nonce > 0 && nonce !== consumedFocusNonce) {
      const resolved = resolveExistingImageId(imageId)
      if (resolved) {
        zoomItemId = resolved
        zoomSliderDrafts = {}
        zoomError = ''
      }
      consumedFocusNonce = nonce
    }
  }
  function closeZoom() { zoomItemId = null; zoomSliderDrafts = {}; zoomError = ''; suppressUntil = performance.now() + 250 }

  // Drag and drop handlers for X/Y axis selectors
  function allowDrop(e) { e.preventDefault(); e.dataTransfer.dropEffect = 'copy' }
  function viewChoiceDetail(nextX, nextY) {
    return `x:${axisName(nextX)}, y:${axisName(nextY)}`
  }
  function applyAxesChoice(nextX, nextY) {
    selectedX = nextX ? String(nextX) : null
    selectedY = nextY ? String(nextY) : null
    dispatch('axesChange', { selectedX, selectedY })
    emitLog('change view', viewChoiceDetail(selectedX, selectedY))
  }
  function onDropX(e) {
    e.preventDefault()
    const id = e.dataTransfer.getData('application/axis-id') || e.dataTransfer.getData('text/plain')
    if (!id) return
    applyAxesChoice(id, selectedY)
  }
  function onDropY(e) {
    e.preventDefault()
    const id = e.dataTransfer.getData('application/axis-id') || e.dataTransfer.getData('text/plain')
    if (!id) return
    applyAxesChoice(selectedX, id)
  }
  function clearX() { applyAxesChoice(null, selectedY) }
  function clearY() { applyAxesChoice(selectedX, null) }
  function adjustThumbSize(delta) {
    cellPx = clamp(Math.round(Number(cellPx || 16) + Number(delta || 0)), MIN_CELL_PX, MAX_CELL_PX)
  }

  // Selections provided by parent and top-center filtering controls
  export let selectionToolsEnabled = true
  export let selections = [] // [{id,name,posIds,negIds,active}]
  export let histogramSlice = null // legacy single-slice shape: { axisId, axisName, startBin, endBin, binCount, ids }
  export let histogramSlices = [] // multi-slice shape: [{ axisId, axisName, startBin, endBin, binCount, ids }]
  $: selectionsById = new Map((selections||[]).map(s => [s.id, s]))
  function normalizeSliceEntry(raw) {
    if (!raw || typeof raw !== 'object') return null
    const axisId = String(raw.axisId || '').trim()
    if (!axisId) return null
    const ids = Array.isArray(raw.ids) ? raw.ids.map((id) => String(id || '').trim()).filter(Boolean) : []
    return { ...raw, axisId, ids }
  }
  $: activeHistogramSlices = (() => {
    const out = []
    const seen = new Set()
    if (Array.isArray(histogramSlices)) {
      for (const raw of histogramSlices) {
        const normalized = normalizeSliceEntry(raw)
        if (!normalized || seen.has(normalized.axisId)) continue
        seen.add(normalized.axisId)
        out.push(normalized)
      }
    }
    if (out.length === 0) {
      const single = normalizeSliceEntry(histogramSlice)
      if (single) out.push(single)
    }
    return out
  })()
  $: histogramSliceActive = activeHistogramSlices.length > 0
  $: activeSubsetChips = subsetChipsFromState(activeHistogramSlices, activeSubsetFilters)
  $: histogramSliceIds = (() => {
    if (!histogramSliceActive) return new Set()
    let intersection = null
    for (const slice of activeHistogramSlices) {
      const ids = new Set(Array.isArray(slice?.ids) ? slice.ids.map((id) => String(id || '').trim()).filter(Boolean) : [])
      if (intersection === null) {
        intersection = ids
      } else {
        intersection = new Set(Array.from(intersection).filter((id) => ids.has(id)))
      }
    }
    return intersection || new Set()
  })()
  function isSliceFilteredOut(id) {
    if (!histogramSliceActive) return false
    return !histogramSliceIds.has(id)
  }
  // Top-center filter dropdown + drag-and-drop of selections
  // Modes: 'all' shows everything; 'keep-pos' shows only selected selection's positives;
  //        'discard-neg' hides selected selection's negatives (keeps non-negative images)
  let filterMode = 'all' // 'all' | 'keep-pos' | 'discard-neg'
  let filterSelection = null // { id, name, posIds:[], negIds:[] }
  let selectedSelectionId = ''
  $: filterPosSet = new Set(Array.isArray(filterSelection?.posIds) ? filterSelection.posIds : [])
  $: filterNegSet = new Set(Array.isArray(filterSelection?.negIds) ? filterSelection.negIds : [])
  // Keep filter in sync with current selections list (e.g., if deleted)
  $: if (filterSelection && selectionsById.size > 0) {
    if (selectionsById.has(filterSelection.id)) {
      // Update reference to the canonical selection object
      filterSelection = selectionsById.get(filterSelection.id)
    } else if (selectedSelectionId) {
      // Selected id is no longer present
      selectedSelectionId = ''
      filterSelection = null
      filterMode = 'all'
    }
  }
  function onFilterModeChange(val) {
    // If no selection is set, non-'all' modes are not applicable
    if ((val === 'keep-pos' || val === 'discard-neg') && !filterSelection) {
      filterMode = 'all'
      return
    }
    // Switching manually clears selection details when returning to 'all'
    filterMode = val
    if (filterMode === 'all') filterSelection = null
  }
  function allowDropSelection(e) { e.preventDefault(); e.dataTransfer.dropEffect = 'copy' }
  function onDropSelection(e) {
    e.preventDefault()
    try {
      const raw = e.dataTransfer.getData('application/x-selection') || e.dataTransfer.getData('text/plain')
      if (!raw) return
      const sel = JSON.parse(raw)
      // Prefer the canonical selection from props when available
      const viaId = sel?.id && selectionsById.get(sel.id)
      const s = viaId || sel
      selectedSelectionId = viaId ? viaId.id : ''
      applySelectionChoice(s)
    } catch (_) { /* ignore */ }
  }
  function onDropdownSelect(id) {
    selectedSelectionId = id || ''
    if (!id) {
      filterSelection = null
      filterMode = 'all'
      return
    }
    const sel = selectionsById.get(id)
    applySelectionChoice(sel)
  }
  function applySelectionChoice(sel) {
    if (!sel) { filterSelection = null; filterMode = 'all'; return }
    const pos = Array.isArray(sel?.posIds) ? sel.posIds : []
    const neg = Array.isArray(sel?.negIds) ? sel.negIds : []
    filterSelection = sel
    if (pos.length > 0 && neg.length === 0) {
      filterMode = 'keep-pos'
    } else if (neg.length > 0 && pos.length === 0) {
      filterMode = 'discard-neg'
    } else if (pos.length > 0 && neg.length > 0) {
      // Mixed: wait for explicit button click; show controls
      filterMode = 'all'
    } else {
      // Empty selection: show all
      filterMode = 'all'
    }
  }

  // Items used for rendering after applying filter.
  $: itemsFiltered = (function() {
    const base = Array.isArray(items) ? items : []
    let out = base
    if (histogramSliceActive) out = out.filter((it) => histogramSliceIds.has(String(it?.id || '')))
    for (const filter of activeSubsetFilters) {
      const ids = new Set(Array.isArray(filter?.ids) ? filter.ids.map((id) => String(id || '').trim()).filter(Boolean) : [])
      if (filter?.mode === 'exclude') out = out.filter((it) => !ids.has(String(it?.id || '')))
      else out = out.filter((it) => ids.has(String(it?.id || '')))
    }
    if (filterMode === 'keep-pos') out = out.filter((it) => filterPosSet.has(it.id))
    else if (filterMode === 'discard-neg') out = out.filter((it) => !filterNegSet.has(it.id))
    return out
  })()
  $: if (zoomItemId && !itemsFiltered.some((it) => String(it?.id || '') === String(zoomItemId))) {
    closeZoom()
  }
</script>

<div class="minimap-shell">
  <div
    role="img"
    aria-label="Axes minimap"
    class="minimap"
    bind:this={minimapEl}
    style={`width:${width}px;height:${height}px;cursor:${grabSaving ? 'progress' : (grabDrag ? 'grabbing' : (lassoEnabled ? 'crosshair' : (grabMode ? 'grab' : 'default')))};`}
    on:pointerdown={onGrabStart}
    on:mousemove={onMove}
    on:wheel|stopPropagation|preventDefault={onWheel}
    on:click|stopPropagation|preventDefault={(e) => pickOrToggle(e)}
  >
  <!-- Density overlay canvas (behind images) -->
  <canvas bind:this={densityCanvas} style="position:absolute;left:0;top:0;z-index:0;opacity:0.9;pointer-events:none;"></canvas>
  <!-- Points overlay canvas (above images) -->
  <canvas bind:this={pointsCanvas} style="position:absolute;left:0;top:0;z-index:12;pointer-events:none;"></canvas>
  {#if activeSubsetChips.length > 0}
    <div class="minimap-chip-stack">
      <button
        type="button"
        class="btn btn-ui-secondary btn-xs minimap-subset-save"
        title="Save current subset"
        on:click|stopPropagation|preventDefault={() => dispatch('saveSubset')}
      >Save subset</button>
      {#each activeSubsetChips as chip (chip.id)}
        <div class="minimap-status-chip minimap-status-chip-subset">
          <span>{chip.label}</span>
          <button
            type="button"
            class="subset-chip-clear"
            aria-label={`Remove ${chip.label}`}
            title={`Remove ${chip.label}`}
            on:click|stopPropagation|preventDefault={() => {
              if (chip.chipType === 'slice') removeHistogramSlice(chip.axisId, chip.label)
              else removeSubsetFilter(chip.id, chip.label)
            }}
          ><MaterialIcon name="close" size={14} /></button>
        </div>
      {/each}
    </div>
  {/if}
  {#if griddingActive && gridBackdrop}
    <div
      class="absolute pointer-events-none rounded"
      style={`left:${gridBackdrop.x0 * 100}%;top:${gridBackdrop.y0 * 100}%;width:${gridBackdrop.w * 100}%;height:${gridBackdrop.h * 100}%;background:rgba(37,99,235,0.16);border:1px solid rgba(37,99,235,0.34);z-index:5; transition: opacity 200ms ease; opacity:1;`}
    />
    {/if}
  {#if pointRadius === 0}
    {#each renderItemsVisible as it (it.id)}
      {@const inside = griddingActive && insideIds.has(it.id)}
      {@const showThumb = showsImageThumb(it.id, inside)}
      {@const thumbPx = renderThumbSizePx(it.id, inside, sizeInside, sizeOutside, subsampleDotPx, subsampleActive)}
      {@const subsetSelected = lassoEnabled && subsetSelectionSet.has(String(it.id || ''))}
      {@const metadataColor = metadataColorForId(it.id)}
      {#if showThumb}
        <img
          alt=""
          src={(inside ? (it.fullUrl || it.url) : (it.thumbUrl || it.url))}
          class="absolute object-cover rounded"
          decoding="async"
          fetchpriority="low"
          style={`left:${it.x * 100}%;
                  top:${(1 - it.y) * 100}%;
                  transform:translate(-50%,-50%);
                  width:${thumbPx}px;
                  height:${thumbPx}px;
                  transition:width 120ms ease,height 120ms ease; 
                  z-index:${isSliceFilteredOut(it.id) ? 0 : (inside ? 10 : 1)}; 
                  opacity:${isSliceFilteredOut(it.id) ? 0.22 : (inside ? 1 : 0.85)}; 
                  filter:${isSliceFilteredOut(it.id) ? 'grayscale(1) saturate(0.12) brightness(1.06)' : 'none'};
                  outline:${metadataColor ? `2px solid ${metadataColor.outlineColor}` : 'none'};
                  outline-offset:${metadataColor ? '1px' : '0'};
                  border:${labelOf(it.id)?'2px solid '+(labelOf(it.id)==='pos'?'#16a34a':'#dc2626'):(subsetSelected ? '2px solid #2563eb' : 'none')}; 
                  box-shadow:${labelOf(it.id)?'0 0 0 1px rgba(255,255,255,0.8)':(subsetSelected ? '0 0 0 1px rgba(255,255,255,0.92)' : 'none')};`}
        />
      {:else}
        <div
          class="absolute subsample-dot"
          aria-hidden="true"
          style={`left:${it.x * 100}%;
                  top:${(1 - it.y) * 100}%;
                  transform:translate(-50%,-50%);
                  width:${thumbPx}px;
                  height:${thumbPx}px;
                  z-index:${isSliceFilteredOut(it.id) ? 0 : 1};
                  opacity:${isSliceFilteredOut(it.id) ? 0.22 : 0.8};
                  background:${metadataColor ? metadataColor.pointColorMuted : '#8f97aa'};
                  box-shadow:${subsetSelected ? '0 0 0 2px rgba(37,99,235,0.9)' : (metadataColor ? `0 0 0 1px ${metadataColor.outlineColor}` : '0 0 0 1px rgba(255, 255, 255, 0.55)')};`}
        />
      {/if}
    {/each}
  {/if}

  {#if !lassoEnabled && !grabMode}
    <div
      bind:this={focusRectEl}
      class="absolute border border-blue-500/70 pointer-events-none"
      style="left:0%;top:0%;width:0%;height:0%;"
    />
  {/if}

  {#if zoomItemId}
    <div class="absolute inset-0 bg-black/40 flex items-center justify-center z-30" on:click|stopPropagation={closeZoom}>
      <div class="zoom-panel" on:click|stopPropagation style="max-width:92%;max-height:88%;">
        <div class="zoom-panel-image-wrap">
          {#if zoomItem?.url}
            <img alt="zoom" src={zoomItem.fullUrl || zoomItem.url} class="zoom-panel-image" />
          {/if}
        </div>
        <div class="zoom-panel-controls">
          {#if zoomAxisEditors.length > 0}
            {#each zoomAxisEditors as editor (editor.axisId)}
              {@const sliderValue = Math.max(0, Math.min(100, Number(zoomSliderDrafts[editor.axisId] ?? editor.currentScore) || 0))}
              <div class="zoom-axis-editor">
                <div class="zoom-axis-editor-head">
                  <span class="zoom-axis-editor-name">{editor.axisName}</span>
                  <span class="zoom-axis-editor-value">{Math.round(sliderValue)}%</span>
                </div>
                <input
                  class="zoom-axis-slider"
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  value={sliderValue}
                  disabled={zoomSavingAxisIds.has(editor.axisId)}
                  on:input={(e) => setZoomDraft(editor.axisId, e.currentTarget.value)}
                  on:change={() => commitZoomAxisEdit(editor.axisId, zoomItemId, editor.currentScore)}
                />
              </div>
            {/each}
          {/if}
          {#if zoomError}
            <div class="zoom-panel-error">{zoomError}</div>
          {/if}
          <div class="mt-2 flex gap-2 justify-center z-40 relative">
            <button class="btn btn-sm btn-ui-secondary" style="z-index:41" on:click|stopPropagation|preventDefault={closeZoom}>
              <MaterialIcon name="close" />
            </button>
          </div>
        </div>
      </div>
    </div>
  {/if}

  <button
    type="button"
    class="minimap-resize-handle"
    title="Drag to resize minimap"
    aria-label="Drag to resize minimap"
    on:pointerdown|stopPropagation|preventDefault={onResizeHandleDown}
  />

  <div class="minimap-tools" on:pointerdown|stopPropagation>
    <button
      type="button"
      class={`btn btn-minimap minimap-tool-btn ${(!grabMode && !lassoEnabled) ? 'is-active' : ''}`}
      on:click|stopPropagation|preventDefault={() => { lassoEnabled = false; clearLassoSelection(); grabMode = false; grabDrag = null; stopGrabListeners() }}
      aria-pressed={!grabMode && !lassoEnabled}
      title="Grid"
      aria-label="Grid"
    >
      <MaterialIcon name="arrow_selector_tool" />
      <span class="minimap-tool-label">GRID</span>
    </button>
    <button
      type="button"
      class={`btn btn-minimap minimap-tool-btn ${grabMode ? 'is-active' : ''}`}
      on:click|stopPropagation|preventDefault={() => { lassoEnabled = false; clearLassoSelection(); grabMode = true }}
      aria-pressed={grabMode}
      title="Move"
      aria-label="Move"
    >
      <MaterialIcon name="tune" />
      <span class="minimap-tool-label">MOVE</span>
    </button>
    <button
      type="button"
      class={`btn btn-minimap minimap-tool-btn ${lassoEnabled ? 'is-active' : ''}`}
      on:click|stopPropagation|preventDefault={toggleLassoTool}
      aria-pressed={lassoEnabled}
      title="Select"
      aria-label="Select"
    >
      <MaterialIcon name="lasso_select" />
      <span class="minimap-tool-label">SELECT</span>
    </button>
    {#if lassoEnabled}
      <button
        type="button"
        class="btn btn-minimap minimap-subset-btn"
        disabled={subsetSelectionSet.size === 0}
        on:click|stopPropagation|preventDefault={() => applySubsetFilter('isolate')}
      >Isolate</button>
      <button
        type="button"
        class="btn btn-minimap minimap-subset-btn"
        disabled={subsetSelectionSet.size === 0}
        on:click|stopPropagation|preventDefault={() => applySubsetFilter('exclude')}
      >Exclude</button>
    {/if}
    <button
      type="button"
      class={`btn btn-minimap minimap-tool-btn ${showDensity ? 'is-active' : ''}`}
      on:click|stopPropagation|preventDefault={() => { showDensity = !showDensity }}
      aria-pressed={showDensity}
      title="Distribution"
      aria-label="Distribution"
    >
      <svg class="minimap-tool-svg density-icon" viewBox="0 0 24 24" aria-hidden="true">
        <path d="M2 18C5 18 5.5 7 12 7s7 11 10 11" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
      </svg>
      <span class="minimap-tool-label">DISTRIBUTION</span>
    </button>
    <label class="minimap-tool-max" title="Maximum number of thumbnails in minimap (0 = all)">
      <span>max</span>
      <input
        class="chip-input"
        type="number"
        min="0"
        step="50"
        value={imageMax}
        on:click|stopPropagation
        on:mousedown|stopPropagation
        on:input={(e)=>{ imageMax = Math.max(0, Math.floor(Number(e.currentTarget.value) || 0)) }}
      />
    </label>
    <div class="minimap-tool-size" title="Thumbnail size">
      <span>size</span>
      <button
        type="button"
        class="btn btn-minimap minimap-size-step"
        on:click|stopPropagation|preventDefault={() => adjustThumbSize(-CELL_PX_STEP)}
        aria-label="Decrease thumbnail size"
        title="Decrease thumbnail size"
      >-</button>
      <span class="minimap-tool-size-value">{cellPx}</span>
      <button
        type="button"
        class="btn btn-minimap minimap-size-step"
        on:click|stopPropagation|preventDefault={() => adjustThumbSize(CELL_PX_STEP)}
        aria-label="Increase thumbnail size"
        title="Increase thumbnail size"
      >+</button>
    </div>
    <label class="minimap-tool-color" title="Color thumbnails and scatter points by metadata axis">
      <span>color</span>
      <select
        class="axis-inline-select minimap-tool-select"
        on:click|stopPropagation
        on:mousedown|stopPropagation
        on:change={(e) => { selectedColorAxisId = String(e.currentTarget.value || '').trim() || null }}
      >
        <option value="" selected={!selectedColorAxisId}>none</option>
        {#each metadataAxes as ax}
          <option value={ax.id} selected={selectedColorAxisId===ax.id}>{ax.name}</option>
        {/each}
      </select>
    </label>
    <button
      type="button"
      class="btn btn-icon btn-minimap minimap-tool-btn"
      disabled={saveVisualizationDisabled}
      on:click|stopPropagation|preventDefault={() => dispatch('saveVisualization')}
      title="Save visualization"
      aria-label="Save visualization"
    ><MaterialIcon name="save" /></button>
    <button type="button" class="btn btn-icon btn-minimap minimap-tool-btn" on:click|stopPropagation|preventDefault={resetView} title="Reset view" aria-label="Reset view"><MaterialIcon name="replay" /></button>
  </div>

  <!-- Axis categorical labels when metadata axes are selected -->
  {#if selectedY && (selectedY.startsWith('axis:meta:'))}
    {#await Promise.resolve(axisTicks(selectedY)) then t}
      {#if t.entries.length > 0}
        {#each t.entries as entry (entry.key)}
          <div class="axis-label y-meta" style={`position:absolute;left:60px;top:${(1 - toVisY(entry.pos)) * 100}%;transform:translateY(-50%);pointer-events:none;font-size:11px;color:#334155;background:rgba(255,255,255,0.85);padding:1px 4px;border-radius:4px;z-index:15;`}>{entry.label}</div>
        {/each}
      {/if}
    {/await}
  {/if}
  {#if selectedX && (selectedX.startsWith('axis:meta:'))}
    {#await Promise.resolve(axisTicks(selectedX)) then t}
      {#if t.entries.length > 0}
        {#each t.entries as entry (entry.key)}
          <div class="axis-label x-meta" style={`position:absolute;bottom:80px;left:${toVisX(entry.pos) * 100}%;transform:translateX(-50%) rotate(-90deg);transform-origin:center;pointer-events:none;font-size:11px;color:#334155;background:rgba(255,255,255,0.85);padding:1px 4px;border-radius:4px;z-index:15;`}>{entry.label}</div>
        {/each}
      {/if}
    {/await}
  {/if}

  {#if selectionToolsEnabled}
    <!-- Top-center filter dropdown and drop target -->
    <div class="axis-rail-top text-sm" role="group" on:dragover={allowDropSelection} on:drop={onDropSelection} title="Drop a selection here or choose one to filter">
      <div class="inline-flex items-center gap-2">
        <span class="text-gray-700 text-sm">Filter</span>
        <select class="text-sm" on:change={(e)=> onDropdownSelect(e.currentTarget.value)}>
          <option value="" selected={!selectedSelectionId}>All images</option>
          {#each (selections||[]) as s}
            <option value={s.id} selected={selectedSelectionId===s.id}>{s.name || s.id}</option>
          {/each}
        </select>
        {#if filterSelection && (Array.isArray(filterSelection.posIds) && filterSelection.posIds.length>0) && (Array.isArray(filterSelection.negIds) && filterSelection.negIds.length>0)}
          <button class="btn btn-xs btn-positive" on:click={() => { filterMode='keep-pos' }} title="Keep only positive examples">Keep positive</button>
          <button class="btn btn-xs btn-negative" on:click={() => { filterMode='discard-neg' }} title="Remove negative examples">Remove negatives</button>
        {/if}
        {#if filterMode!=='all' && filterSelection}
          <button class="btn btn-xs btn-ui-secondary" on:click={() => { filterMode='all'; filterSelection=null; selectedSelectionId='' }}>Clear</button>
        {/if}
      </div>
    </div>
  {/if}

  <div
    class="axis-edge axis-edge-y text-sm"
    role="group"
    style={`left:${axisFrameLeftPx + Y_AXIS_EDGE_NUDGE_PX}px;top:${axisFrameCenterYPx}px;`}
    on:dragover={allowDrop}
    on:drop={onDropY}
    title="Drop a Y axis here"
  >
    <select class="axis-inline-select axis-inline-select-y text-sm" on:change={(e)=>{ applyAxesChoice(selectedX, e.currentTarget.value || null) }}>
      <option value="">(none)</option>
      {#each axes as ax}
        <option value={ax.id} selected={selectedY===ax.id}>{ax.name}</option>
      {/each}
    </select>
  </div>

  {#if selectionToolsEnabled}
    <!-- Bottom-right create selection button -->
    <div class="toolbar pos-bottom-right right-just z-10">
      <button class="btn btn-sm btn-minimap" on:click|stopPropagation={() => {
        const pos = []
        const neg = []
        try {
          if (labels instanceof Map) { labels.forEach((v,k)=>{ if (v==='pos') pos.push(k); else if (v==='neg') neg.push(k) }) }
          else { for (const [k,v] of Object.entries(labels||{})) { if (v==='pos') pos.push(k); else if (v==='neg') neg.push(k) } }
        } catch(_) {}
        const name = prompt('Name this selection', 'Selection') || 'Selection'
        dispatch('saveSelection', { id: `sel:${Date.now()}`, name, posIds: pos, negIds: neg, active: true })
      }}>Create selection</button>
    </div>
  {/if}

  {#if lassoEnabled}
    <LassoSelector
      bind:this={lassoRef}
      enabled={true}
      width={width}
      height={height}
      items={lassoItems}
      strokeColor="#2563eb"
      fillColor="rgba(37,99,235,0.14)"
      on:select={onLassoSelect}
    />
  {/if}
  <div
    class="axis-edge axis-edge-x text-sm"
    role="group"
    style={`left:${axisFrameCenterXPx}px;top:${axisFrameBottomPx - 5}px;`}
    on:dragover={allowDrop}
    on:drop={onDropX}
    title="Drop an X axis here"
  >
      <select class="axis-inline-select text-sm" on:change={(e)=>{ applyAxesChoice(e.currentTarget.value || null, selectedY) }}>
        <option value="">(none)</option>
        {#each axes as ax}
          <option value={ax.id} selected={selectedX===ax.id}>{ax.name}</option>
        {/each}
      </select>
    </div>
  </div>
</div>

<style>
  .minimap-shell {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: flex-start;
    overflow: visible;
  }

  .zoom-panel {
    background: #ffffff;
    border-radius: 8px;
    box-shadow: 0 20px 50px rgba(15, 23, 42, 0.24);
    padding: 14px;
    display: grid;
    gap: 12px;
    width: min(620px, 92vw);
  }

  .zoom-panel-image-wrap {
    display: grid;
    place-items: center;
  }

  .zoom-panel-image {
    width: min(560px, 86vw);
    max-height: 62vh;
    object-fit: contain;
    display: block;
    border-radius: 6px;
    background: #f8fafc;
  }

  .zoom-panel-controls {
    display: grid;
    gap: 10px;
  }

  .zoom-axis-editor {
    display: grid;
    gap: 6px;
  }

  .zoom-axis-editor-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    font-size: var(--font-size-body);
    color: #334155;
  }

  .zoom-axis-editor-name {
    font-weight: 600;
  }

  .zoom-axis-editor-value {
    color: #64748b;
  }

  .zoom-axis-slider {
    width: 100%;
  }

  .zoom-panel-error {
    font-size: var(--font-size-small);
    color: #dc2626;
  }

  .subsample-dot {
    border-radius: 999px;
    background: #8f97aa;
    box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.55);
  }

  .minimap-chip-stack {
    position: absolute;
    top: 8px;
    right: 8px;
    z-index: 30;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 6px;
    max-width: min(320px, calc(100% - 16px));
  }

  .minimap-status-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    min-height: 28px;
    padding: 4px 8px;
    border-radius: 999px;
    border: 1px solid #dbe2ec;
    background: rgba(255, 255, 255, 0.95);
    color: #334155;
    font-size: 12px;
    line-height: 1;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
    max-width: 100%;
  }

  .minimap-status-chip-subset {
    font-weight: 600;
  }

  .minimap-subset-save {
    height: 28px;
    padding: 0 10px;
    border-radius: 999px;
    white-space: nowrap;
    align-self: flex-end;
  }

  .subset-chip-clear {
    width: 18px;
    height: 18px;
    padding: 0;
    border: 0;
    border-radius: 999px;
    background: transparent;
    color: #64748b;
    font-size: 14px;
    line-height: 1;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }

  .axis-edge {
    position: absolute;
    z-index: 24;
    display: inline-flex;
    align-items: center;
    gap: 0;
    color: #5b6472;
    pointer-events: auto;
  }

  .axis-edge-y {
    transform: translate(-100%, -50%);
    flex-direction: row;
    align-items: center;
    justify-content: center;
    min-width: 0;
    min-height: 0;
  }

  .axis-edge-x {
    transform: translate(-50%, -100%);
    align-items: center;
  }

  .axis-inline-select {
    margin: 0;
    width: auto;
    max-width: none;
    border: 0;
    border-bottom: 1px solid #d8dee8;
    border-radius: 0;
    background: transparent;
    color: #334155;
    padding: 2px 20px 2px 4px;
    line-height: 1.15;
    box-shadow: none;
  }

  .axis-inline-select:focus {
    outline: none;
    border-bottom-color: #94a3b8;
  }

  .axis-inline-select-y {
    width: auto;
    min-width: max-content;
    transform: rotate(-90deg);
    transform-origin: center center;
  }

  .minimap-resize-handle {
    position: absolute;
    right: 6px;
    bottom: 44px;
    width: 14px;
    height: 14px;
    margin: 0;
    padding: 0;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    background: #ffffff;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.16);
    cursor: nwse-resize;
    z-index: 32;
    touch-action: none;
  }

  .minimap-resize-handle:hover {
    border-color: #94a3b8;
    background: #f8fafc;
  }

  .minimap-tools {
    position: absolute;
    top: 8px;
    left: 8px;
    z-index: 30;
    pointer-events: auto;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    white-space: nowrap;
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid #dbe2ec;
    border-radius: 6px;
    padding: 4px 6px;
  }

  .minimap-tool-btn {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    color: #0f172a !important;
    border-color: #cbd5e1;
    min-width: 0;
    height: 28px;
    padding: 0 8px;
    opacity: 1;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.03em;
  }

  .minimap-tool-btn.is-active {
    background: rgba(37, 99, 235, 0.1);
    border-color: #2563eb;
    color: #2563eb !important;
  }

  .minimap-tool-btn.is-active :global(.material-symbols-rounded.material-symbol),
  .minimap-tool-btn.is-active .minimap-tool-svg {
    color: #2563eb !important;
  }

  .minimap-subset-btn {
    min-width: 68px;
    height: 28px;
    padding: 0 10px;
    color: #0f172a !important;
    border-color: #cbd5e1;
    background: #ffffff;
    font-size: 12px;
  }

  .minimap-subset-btn:disabled {
    opacity: 0.45;
    cursor: default;
  }

  .minimap-tool-svg {
    width: 16px;
    height: 16px;
    display: block;
    color: #0f172a;
  }

  .minimap-tool-label {
    line-height: 1;
  }

  .density-icon {
    width: 18px;
    height: 18px;
  }

  .minimap-tool-max {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    color: #334155;
    font-size: 11px;
  }

  .minimap-tool-max :global(input) {
    width: 58px;
    height: 26px;
    font-size: 11px;
    padding: 0 4px;
  }

  .minimap-tool-size {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    color: #334155;
    font-size: 11px;
  }

  .minimap-tool-color {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: #334155;
    font-size: 11px;
  }

  .minimap-tool-select {
    min-width: 124px;
    max-width: 180px;
    height: 28px;
  }

  .minimap-tool-size-value {
    min-width: 20px;
    text-align: center;
    color: #475569;
    font-variant-numeric: tabular-nums;
  }

  .minimap-size-step {
    min-width: 24px;
    height: 24px;
    padding: 0;
    border-color: #cbd5e1;
    font-size: 14px;
    line-height: 1;
  }

</style>
