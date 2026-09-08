<script>
  import { createEventDispatcher, onDestroy, onMount } from 'svelte'
  import MaterialIcon from './MaterialIcon.svelte'

  export let session = null
  export let itemsById = new Map()
  export let subsetIds = []
  export let subsetActive = false
  export let selectedX = null
  export let selectedY = null
  export let busy = false
  export let promptUpdateBusy = false
  export let activeSlice = null

  const dispatch = createEventDispatcher()
  const SLICE_SEGMENTS = 24
  const MIN_SLICE_WIDTH_PCT = 10
  const KDE_WIDTH = 500
  const KDE_HEIGHT = 84
  const HISTOGRAM_BIN_COUNT = 10
  const HISTOGRAM_WIDTH = 500
  const HISTOGRAM_HEIGHT = 84
  let dragSlice = null
  let localActiveSlice = null
  let pendingCommittedSlice = null
  let hoveredDecileIndex = null
  let previewDecileIndex = null
  let previewHistogramIndex = null
  let dropDecileIndex = null
  let dropUndefinedActive = false
  let undefinedTooltipOpen = false
  let openPriorPromptSide = null // 'neg' | 'pos' | null
  let editingPriorPromptSide = null
  let promptDraftText = ''
  let feedbackDrag = null
  let hoveredFeedbackId = ''
  let deleteModifierActive = false
  let subsetOnly = false

  function normalizeImageId(v) {
    return String(v || '').trim()
  }

  function basenameId(v) {
    const s = normalizeImageId(v)
    if (!s) return ''
    const parts = s.split(/[\\/]/)
    return parts[parts.length - 1] || s
  }

  function resolveItem(imageId) {
    const direct = itemsById?.get?.(normalizeImageId(imageId))
    if (direct) return direct
    const base = basenameId(imageId)
    if (base && base !== normalizeImageId(imageId)) {
      const byBase = itemsById?.get?.(base)
      if (byBase) return byBase
    }
    if (!itemsById || typeof itemsById.values !== 'function') return null
    for (const item of itemsById.values()) {
      const itemId = normalizeImageId(item?.id)
      if (!itemId) continue
      if (itemId === normalizeImageId(imageId) || basenameId(itemId) === base) return item
    }
    return null
  }

  function clamp01(v) {
    const n = Number(v)
    if (!Number.isFinite(n)) return 0
    if (n < 0) return 0
    if (n > 1) return 1
    return n
  }

  function clamp100(v) {
    return clamp01((Number(v) || 0) / 100) * 100
  }

  $: normalizedSubsetIds = Array.isArray(subsetIds) ? subsetIds.map((id) => String(id || '').trim()).filter(Boolean) : []
  $: subsetIdSet = new Set(normalizedSubsetIds)
  $: if (!subsetActive) subsetOnly = false

  function formatRawValue(v) {
    const n = Number(v)
    if (!Number.isFinite(n)) return '0'
    return (Math.round(n * 10) / 10).toFixed(1)
  }

  function segmentIndexForPct(score) {
    const pct = clamp100(score)
    if (pct >= 100) return SLICE_SEGMENTS - 1
    return Math.max(0, Math.min(SLICE_SEGMENTS - 1, Math.floor((pct / 100) * SLICE_SEGMENTS)))
  }

  function parseDragPayload(e) {
    try {
      const raw = e?.dataTransfer?.getData?.('application/x-axis-builder-item')
      if (!raw) return null
      const parsed = JSON.parse(raw)
      if (!parsed || typeof parsed !== 'object') return null
      return {
        imageId: String(parsed.imageId || ''),
        score: Number(parsed.score || 0),
        sourceRow: String(parsed.sourceRow || '').trim(),
        sourceRank: parsed.sourceRank === undefined || parsed.sourceRank === null ? '' : String(parsed.sourceRank).trim(),
      }
    } catch (_) {
      return null
    }
  }

  function buildDragPayload(item, extras = {}) {
    return {
      imageId: item?.id,
      score: Number(item?.score_0_100 || 0),
      sourceRow: String(extras?.sourceRow || '').trim(),
      sourceRank: extras?.sourceRank === undefined || extras?.sourceRank === null ? '' : String(extras.sourceRank).trim(),
    }
  }

  function allowDrop(e) {
    try {
      e.preventDefault()
      e.dataTransfer.dropEffect = 'move'
    } catch (_) {}
  }

  function onDragStart(e, item, extras = {}) {
    if (!item?.id || busy) return
    try {
      const payload = buildDragPayload(item, extras)
      e.dataTransfer.effectAllowed = 'move'
      e.dataTransfer.setData('application/x-axis-builder-item', JSON.stringify(payload))
      e.dataTransfer.setData('text/plain', String(item.id))
    } catch (_) {}
  }

  function decileTargetPct(index) {
    const idx = Math.max(0, Math.min(9, Number(index) || 0))
    return (idx * 10) + 5
  }

  function dispatchMove(imageId, fromScore0To100, targetPct, moveType = 'score', extra = {}) {
    dispatch('move', {
      axisId: session?.axis?.id || session?.axisId,
      imageId,
      newScore0To100: clamp100(targetPct),
      fromScore0To100: clamp100(fromScore0To100),
      moveType,
      sourceRow: String(extra?.sourceRow || '').trim(),
      sourceRank: extra?.sourceRank === undefined || extra?.sourceRank === null ? '' : String(extra.sourceRank).trim(),
      targetRank: moveType === 'undefined' ? 'undefined' : String(extra?.targetRank || '').trim(),
    })
  }

  function isDeleteModifierKey(key) {
    const normalized = String(key || '').trim().toLowerCase()
    return normalized === 'delete' || normalized === 'backspace'
  }

  function clearFeedbackPointerListeners() {
    window.removeEventListener('pointermove', onWindowFeedbackPointerMove)
    window.removeEventListener('pointerup', onWindowFeedbackPointerUp)
  }

  function clearFeedbackDrag() {
    feedbackDrag = null
    clearFeedbackPointerListeners()
  }

  function feedbackTargetRank(targetPct) {
    const pct = clamp100(targetPct)
    if (pct >= 100) return 10
    return Math.max(1, Math.min(10, Math.floor(pct / 10) + 1))
  }

  function feedbackDisplayPct(entry) {
    const entryId = normalizeImageId(entry?.id)
    if (feedbackDrag && feedbackDrag.imageId === entryId) {
      return clamp100(feedbackDrag.currentPct)
    }
    return clamp100(entry?.score_0_100 || 0)
  }

  function feedbackTrackRect(target) {
    return target?.closest?.('.feedback-track')?.getBoundingClientRect?.() || null
  }

  function onWindowFeedbackKeyDown(e) {
    if (isDeleteModifierKey(e?.key)) deleteModifierActive = true
  }

  function onWindowFeedbackKeyUp(e) {
    if (isDeleteModifierKey(e?.key)) deleteModifierActive = false
  }

  function onWindowFeedbackBlur() {
    deleteModifierActive = false
    clearFeedbackDrag()
  }

  function startFeedbackPointInteraction(e, entry) {
    const imageId = normalizeImageId(entry?.id)
    if (!imageId || busy) return
    if (Number(e?.button ?? 0) !== 0) return
    const currentPct = clamp100(entry?.score_0_100 || 0)
    if (deleteModifierActive) {
      dispatchMove(imageId, currentPct, currentPct, 'delete', {
        sourceRow: 'feedback',
        sourceRank: 'feedback',
        targetRank: 'deleted',
      })
      try { e.preventDefault() } catch (_) {}
      return
    }
    const rect = feedbackTrackRect(e?.currentTarget)
    if (!rect) return
    feedbackDrag = {
      imageId,
      rect,
      fromScore0To100: currentPct,
      currentPct,
      moved: false,
    }
    hoveredFeedbackId = imageId
    window.addEventListener('pointermove', onWindowFeedbackPointerMove)
    window.addEventListener('pointerup', onWindowFeedbackPointerUp, { once: true })
    try { e.preventDefault() } catch (_) {}
  }

  function onWindowFeedbackPointerMove(e) {
    if (!feedbackDrag) return
    const nextPct = clientPct(e, feedbackDrag.rect)
    feedbackDrag = {
      ...feedbackDrag,
      currentPct: nextPct,
      moved: feedbackDrag.moved || Math.abs(nextPct - feedbackDrag.fromScore0To100) >= 0.6,
    }
  }

  function onWindowFeedbackPointerUp(e) {
    if (!feedbackDrag) return
    const active = feedbackDrag
    clearFeedbackPointerListeners()
    const nextPct = clientPct(e, active.rect)
    const didMove = active.moved || Math.abs(nextPct - active.fromScore0To100) >= 0.6
    feedbackDrag = null
    if (!didMove) {
      openImage(active.imageId)
      return
    }
    dispatchMove(active.imageId, active.fromScore0To100, nextPct, 'score', {
      sourceRow: 'feedback',
      sourceRank: 'feedback',
      targetRank: feedbackTargetRank(nextPct),
    })
  }

  function onDropZoneEnter(index) {
    const idx = Math.max(0, Math.min(9, Number(index) || 0))
    hoveredDecileIndex = idx
    dropDecileIndex = idx
    dropUndefinedActive = false
  }

  function onDropZoneLeave(index = null) {
    if (index === null || dropDecileIndex === index) dropDecileIndex = null
  }

  function onDropZone(e, index) {
    const payload = parseDragPayload(e)
    if (!payload?.imageId) return
    const idx = Math.max(0, Math.min(9, Number(index) || 0))
    dropDecileIndex = null
    hoveredDecileIndex = idx
    dispatchMove(payload.imageId, payload.score, decileTargetPct(idx), 'score', {
      sourceRow: payload.sourceRow,
      sourceRank: payload.sourceRank,
      targetRank: idx + 1,
    })
  }

  function onUndefinedDropEnter() {
    dropUndefinedActive = true
  }

  function onUndefinedDropLeave() {
    dropUndefinedActive = false
  }

  function onUndefinedDrop(e) {
    const payload = parseDragPayload(e)
    if (!payload?.imageId) return
    dropUndefinedActive = false
    undefinedTooltipOpen = false
    dispatchMove(payload.imageId, payload.score, payload.score, 'undefined', {
      sourceRow: payload.sourceRow,
      sourceRank: payload.sourceRank || 'undefined',
      targetRank: 'undefined',
    })
  }

  function toggleUndefinedTooltip() {
    undefinedTooltipOpen = !undefinedTooltipOpen
  }

  function closeUndefinedTooltip() {
    undefinedTooltipOpen = false
  }

  function rawThresholdForDisplayPct(displayPct) {
    if (!Number.isFinite(rawMin) || !Number.isFinite(rawMax) || Math.abs(rawMax - rawMin) <= 1e-8) {
      return 0
    }
    const t = clamp100(displayPct) / 100
    return rawMin + (t * (rawMax - rawMin))
  }

  function buildSlice(startDisplayPct, endDisplayPct) {
    const loPct = Math.min(clamp100(startDisplayPct), clamp100(endDisplayPct))
    const hiPct = Math.max(clamp100(startDisplayPct), clamp100(endDisplayPct))
    if ((hiPct - loPct) < MIN_SLICE_WIDTH_PCT) {
      return null
    }
    const rawLo = rawThresholdForDisplayPct(loPct)
    const rawHi = rawThresholdForDisplayPct(hiPct)
    const ids = rawEntries
      .filter((entry) => Number(entry.rawValue || 0) >= rawLo && Number(entry.rawValue || 0) <= rawHi)
      .map((entry) => entry.id)

    return {
      axisId: session?.axis?.id || session?.axisId,
      axisName: session?.axis?.name || session?.q || session?.axisId,
      startBin: segmentIndexForPct(loPct),
      endBin: segmentIndexForPct(hiPct),
      binCount: SLICE_SEGMENTS,
      rangeStartPct: loPct,
      rangeEndPct: hiPct,
      ids,
    }
  }

  function commitSlice(nextSlice) {
    localActiveSlice = nextSlice
    dispatch('sliceChange', { slice: nextSlice || null })
  }

  function commitSliceFinal(nextSlice) {
    localActiveSlice = nextSlice
    pendingCommittedSlice = nextSlice || null
    dispatch('sliceChange', { slice: nextSlice || null, final: true })
  }

  function clearSlice() {
    localActiveSlice = null
    pendingCommittedSlice = null
    dragSlice = null
    stopSlicePointerListeners()
    dispatch('sliceChange', { slice: null })
  }

  function sliceSignature(slice) {
    if (!slice || typeof slice !== 'object') return ''
    const ids = Array.isArray(slice.ids) ? slice.ids.map((id) => String(id || '').trim()).filter(Boolean) : []
    return JSON.stringify({
      axisId: String(slice.axisId || '').trim(),
      rangeStartPct: clamp100(slice.rangeStartPct || 0),
      rangeEndPct: clamp100(slice.rangeEndPct || 0),
      ids,
    })
  }

  function startSlicePointerListeners() {
    stopSlicePointerListeners()
    window.addEventListener('pointermove', onWindowPointerMove)
    window.addEventListener('pointerup', onWindowPointerUp)
  }

  function stopSlicePointerListeners() {
    window.removeEventListener('pointermove', onWindowPointerMove)
    window.removeEventListener('pointerup', onWindowPointerUp)
  }

  function clientPct(e, rect) {
    if (!rect || !Number.isFinite(rect.width) || rect.width <= 0) return 0
    return clamp01((Number(e.clientX || 0) - rect.left) / rect.width) * 100
  }

  function onPlotPointerDown(e) {
    const rect = e.currentTarget?.getBoundingClientRect?.()
    if (!rect) return
    const startDisplayPct = clientPct(e, rect)
    dragSlice = { kind: 'create', rect, startDisplayPct, lastDisplayPct: startDisplayPct }
    startSlicePointerListeners()
    try { e.preventDefault() } catch (_) {}
  }

  function slicePreviewFromDrag() {
    if (!dragSlice) return null
    if (dragSlice.kind === 'resize') {
      return buildSlice(dragSlice.fixedDisplayPct, dragSlice.movingDisplayPct)
    }
    return buildSlice(dragSlice.startDisplayPct, dragSlice.lastDisplayPct)
  }

  function startSliceResize(e, side) {
    if (!localActiveSlice || busy) return
    const shell = e?.currentTarget?.closest?.('.histogram-shell')
    const rect = shell?.getBoundingClientRect?.()
    if (!rect) return
    dragSlice = {
      kind: 'resize',
      rect,
      side: side === 'end' ? 'end' : 'start',
      fixedDisplayPct: side === 'end'
        ? clamp100(localActiveSlice.rangeStartPct || 0)
        : clamp100(localActiveSlice.rangeEndPct || 0),
      movingDisplayPct: side === 'end'
        ? clamp100(localActiveSlice.rangeEndPct || 0)
        : clamp100(localActiveSlice.rangeStartPct || 0),
    }
    startSlicePointerListeners()
    try { e.preventDefault() } catch (_) {}
    try { e.stopPropagation() } catch (_) {}
  }

  function onWindowPointerMove(e) {
    if (!dragSlice) return
    if (dragSlice.kind === 'resize') {
      dragSlice.movingDisplayPct = clientPct(e, dragSlice.rect)
    } else {
      dragSlice.lastDisplayPct = clientPct(e, dragSlice.rect)
    }
    const nextSlice = slicePreviewFromDrag()
    if (nextSlice) commitSlice(nextSlice)
  }

  function onWindowPointerUp() {
    if (dragSlice) {
      const nextSlice = slicePreviewFromDrag()
      if (nextSlice) commitSliceFinal(nextSlice)
    }
    dragSlice = null
    stopSlicePointerListeners()
  }

  function buildDensityPlot(values, width = KDE_WIDTH, height = KDE_HEIGHT) {
    const baseY = height - 10
    if (!Array.isArray(values) || values.length === 0 || Math.abs(rawMax - rawMin) <= 1e-8) {
      return {
        width,
        height,
        baseY,
        linePath: `M 0 ${baseY} L ${width} ${baseY}`,
        points: [
          { xPct: 0, x: 0, y: baseY },
          { xPct: 100, x: width, y: baseY },
        ],
      }
    }

    const normalized = values
      .map((value) => clamp01((Number(value || 0) - rawMin) / Math.max(1e-8, rawMax - rawMin)) * 100)
      .filter((value) => Number.isFinite(value))
    const n = Math.max(1, normalized.length)
    const steps = Math.max(72, Math.min(200, Math.round(n * 1.8)))
    const bandwidth = Math.max(2.4, Math.min(10, 70 / Math.sqrt(n + 6)))
    const rawPoints = []
    let maxDensity = 1e-6

    for (let i = 0; i < steps; i += 1) {
      const xPct = (i / Math.max(1, steps - 1)) * 100
      let density = 0
      for (const value of normalized) {
        const z = (xPct - value) / bandwidth
        density += Math.exp(-0.5 * z * z)
      }
      if (density > maxDensity) maxDensity = density
      rawPoints.push({ xPct, density })
    }

    const usableHeight = height - 20
    const points = rawPoints.map((pt) => {
      const x = (pt.xPct / 100) * width
      const y = baseY - ((pt.density / maxDensity) * usableHeight)
      return { xPct: pt.xPct, x, y }
    })
    const linePath = points
      .map((pt, idx) => `${idx === 0 ? 'M' : 'L'} ${pt.x.toFixed(2)} ${pt.y.toFixed(2)}`)
      .join(' ')

    return { width, height, baseY, linePath, points }
  }

  function buildUncertaintyTrend(rawValues, stdValues, width = KDE_WIDTH, height = KDE_HEIGHT) {
    const n = Math.min(
      Array.isArray(rawValues) ? rawValues.length : 0,
      Array.isArray(stdValues) ? stdValues.length : 0,
    )
    if (n <= 3 || !Number.isFinite(rawMin) || !Number.isFinite(rawMax) || Math.abs(rawMax - rawMin) <= 1e-8) {
      return { meanPath: '', bandPath: '' }
    }

    const pairs = []
    for (let i = 0; i < n; i += 1) {
      const raw = Number(rawValues[i])
      const std = Number(stdValues[i])
      if (!Number.isFinite(raw) || !Number.isFinite(std)) continue
      pairs.push({ raw, std })
    }
    if (pairs.length <= 3) return { meanPath: '', bandPath: '' }

    const binCount = Math.max(18, Math.min(40, Math.round(Math.sqrt(pairs.length) * 2.5)))
    const bins = Array.from({ length: binCount }, () => [])
    const span = Math.max(1e-8, rawMax - rawMin)
    for (const p of pairs) {
      const ratio = clamp01((p.raw - rawMin) / span)
      const idx = Math.min(binCount - 1, Math.max(0, Math.floor(ratio * binCount)))
      bins[idx].push(p.std)
    }

    const means = new Array(binCount).fill(null)
    const sds = new Array(binCount).fill(0)
    for (let i = 0; i < binCount; i += 1) {
      const arr = bins[i]
      if (!arr || arr.length === 0) continue
      const m = arr.reduce((acc, v) => acc + v, 0) / arr.length
      let varSum = 0
      for (const v of arr) {
        const dv = v - m
        varSum += dv * dv
      }
      means[i] = m
      sds[i] = Math.sqrt(varSum / Math.max(1, arr.length))
    }

    // Fill empty bins by interpolating to avoid visual gaps.
    for (let i = 0; i < binCount; i += 1) {
      if (means[i] !== null) continue
      let left = i - 1
      while (left >= 0 && means[left] === null) left -= 1
      let right = i + 1
      while (right < binCount && means[right] === null) right += 1
      if (left >= 0 && right < binCount) {
        const t = (i - left) / Math.max(1, right - left)
        means[i] = Number(means[left]) + ((Number(means[right]) - Number(means[left])) * t)
        sds[i] = Number(sds[left]) + ((Number(sds[right]) - Number(sds[left])) * t)
      } else if (left >= 0) {
        means[i] = Number(means[left])
        sds[i] = Number(sds[left])
      } else if (right < binCount) {
        means[i] = Number(means[right])
        sds[i] = Number(sds[right])
      }
    }

    const valid = means
      .map((m, idx) => ({ idx, mean: Number(m), sd: Number(sds[idx] || 0) }))
      .filter((row) => Number.isFinite(row.mean))
    if (valid.length <= 2) return { meanPath: '', bandPath: '' }

    let statsMin = Infinity
    let statsMax = -Infinity
    for (const row of valid) {
      const lo = row.mean - row.sd
      const hi = row.mean + row.sd
      if (lo < statsMin) statsMin = lo
      if (hi > statsMax) statsMax = hi
    }
    if (!Number.isFinite(statsMin) || !Number.isFinite(statsMax) || Math.abs(statsMax - statsMin) <= 1e-8) {
      statsMin = 0
      statsMax = 1
    }
    const valueSpan = Math.max(1e-8, statsMax - statsMin)
    const baseY = height - 10
    const usableHeight = height - 20
    const xForBin = (idx) => ((idx + 0.5) / binCount) * width
    const yForValue = (value) => baseY - (((value - statsMin) / valueSpan) * usableHeight)

    const upper = []
    const lower = []
    const mean = []
    for (const row of valid) {
      const x = xForBin(row.idx)
      const yMean = yForValue(row.mean)
      const yUpper = yForValue(row.mean + row.sd)
      const yLower = yForValue(row.mean - row.sd)
      mean.push({ x, y: yMean })
      upper.push({ x, y: yUpper })
      lower.push({ x, y: yLower })
    }

    const meanPath = mean
      .map((pt, idx) => `${idx === 0 ? 'M' : 'L'} ${pt.x.toFixed(2)} ${pt.y.toFixed(2)}`)
      .join(' ')
    const bandPath = [
      ...upper.map((pt, idx) => `${idx === 0 ? 'M' : 'L'} ${pt.x.toFixed(2)} ${pt.y.toFixed(2)}`),
      ...lower.slice().reverse().map((pt) => `L ${pt.x.toFixed(2)} ${pt.y.toFixed(2)}`),
      'Z',
    ].join(' ')
    return { meanPath, bandPath }
  }

  function quantileSorted(values, q) {
    if (!Array.isArray(values) || values.length === 0) return 0
    const t = clamp01(q)
    if (values.length === 1) return Number(values[0] || 0)
    const pos = t * (values.length - 1)
    const lo = Math.floor(pos)
    const hi = Math.min(values.length - 1, Math.ceil(pos))
    if (lo === hi) return Number(values[lo] || 0)
    const mix = pos - lo
    return (Number(values[lo] || 0) * (1 - mix)) + (Number(values[hi] || 0) * mix)
  }

  function xPctForRawValue(value) {
    if (!Number.isFinite(rawMin) || !Number.isFinite(rawMax) || Math.abs(rawMax - rawMin) <= 1e-8) return 50
    return clamp01((Number(value || 0) - rawMin) / Math.max(1e-8, rawMax - rawMin)) * 100
  }

  function yForXPct(points, xPct, fallbackY) {
    if (!Array.isArray(points) || points.length === 0) return fallbackY
    const target = clamp100(xPct)
    let prev = points[0]
    for (let i = 1; i < points.length; i += 1) {
      const next = points[i]
      if (target <= next.xPct) {
        const span = Math.max(1e-6, next.xPct - prev.xPct)
        const t = (target - prev.xPct) / span
        return prev.y + ((next.y - prev.y) * t)
      }
      prev = next
    }
    return points[points.length - 1]?.y ?? fallbackY
  }

  function decileColumns(deciles, activeSubsetIds = null) {
    const rows = Array.isArray(deciles) ? deciles : []
    const allowed = activeSubsetIds instanceof Set ? activeSubsetIds : null
    return Array.from({ length: 10 }, (_, idx) => {
      const row = rows.find((entry) => Number(entry?.bin_index || 0) === idx) || {}
      const rawExemplars = Array.isArray(row?.exemplars) ? row.exemplars : []
      const filteredExemplars = allowed
        ? rawExemplars.filter((entry) => allowed.has(String(entry?.id || '').trim()))
        : rawExemplars
      const sortedExemplars = [...filteredExemplars].sort((a, b) => (Number(a?.std) || 0) - (Number(b?.std) || 0))
      const mostCertain = sortedExemplars.length > 0 ? sortedExemplars[0] : null
      const leastCertain = sortedExemplars.length > 0 ? sortedExemplars[sortedExemplars.length - 1] : null
      return {
        index: idx,
        label: `${idx * 10}-${(idx + 1) * 10}%`,
        mostCertain,
        leastCertain,
        samples: sortedExemplars.slice(0, 4),
      }
    })
  }

  function buildHistogramBins(entries, width = HISTOGRAM_WIDTH, height = HISTOGRAM_HEIGHT, binCount = HISTOGRAM_BIN_COUNT) {
    const safeCount = Math.max(1, Number(binCount) || 1)
    const baseY = height
    const usableHeight = height - 4
    const gap = 0
    const totalGap = gap * Math.max(0, safeCount - 1)
    const barWidth = Math.max(10, (width - totalGap) / safeCount)
    const span = Math.max(1e-8, rawMax - rawMin)
    const bins = Array.from({ length: safeCount }, (_, idx) => ({
      index: idx,
      entries: [],
      count: 0,
      centerRaw: rawMin + (((idx + 0.5) / safeCount) * span),
    }))

    for (const entry of Array.isArray(entries) ? entries : []) {
      const raw = Number(entry?.rawValue || 0)
      if (!Number.isFinite(raw)) continue
      const ratio = clamp01((raw - rawMin) / span)
      const idx = Math.min(safeCount - 1, Math.max(0, Math.floor(ratio * safeCount)))
      bins[idx].entries.push(entry)
    }

    let maxCount = 1
    for (const bin of bins) {
      bin.count = bin.entries.length
      if (bin.count > maxCount) maxCount = bin.count
    }

    return bins.map((bin, idx) => {
      const x = idx * (barWidth + gap)
      const h = Math.max(bin.count > 0 ? 8 : 3, (bin.count / maxCount) * usableHeight)
      const y = baseY - h
      const samples = [...bin.entries]
        .sort((a, b) => {
          const da = Math.abs(Number(a?.rawValue || 0) - bin.centerRaw)
          const db = Math.abs(Number(b?.rawValue || 0) - bin.centerRaw)
          if (Math.abs(da - db) > 1e-8) return da - db
          return (Number(a?.std || 0) || 0) - (Number(b?.std || 0) || 0)
        })
        .slice(0, 4)
      return {
        ...bin,
        x,
        y,
        width: barWidth,
        height: h,
        centerPct: ((x + (barWidth * 0.5)) / width) * 100,
        samples,
      }
    })
  }

  function onDecileEnter(index) {
    hoveredDecileIndex = Math.max(0, Math.min(9, Number(index) || 0))
  }

  function onDecileLeave(index) {
    const idx = Math.max(0, Math.min(9, Number(index) || 0))
    if (hoveredDecileIndex === idx) hoveredDecileIndex = null
    onDropZoneLeave(idx)
  }

  function onMarkerEnter(index) {
    const idx = Math.max(0, Math.min(9, Number(index) || 0))
    hoveredDecileIndex = idx
    previewDecileIndex = idx
    previewHistogramIndex = null
  }

  function onMarkerLeave(index) {
    const idx = Math.max(0, Math.min(9, Number(index) || 0))
    if (previewDecileIndex === idx) previewDecileIndex = null
    onDecileLeave(idx)
  }

  function onHistogramEnter(index) {
    const idx = Math.max(0, Math.min(HISTOGRAM_BIN_COUNT - 1, Number(index) || 0))
    previewHistogramIndex = idx
    previewDecileIndex = null
  }

  function onHistogramLeave(index) {
    const idx = Math.max(0, Math.min(HISTOGRAM_BIN_COUNT - 1, Number(index) || 0))
    if (previewHistogramIndex === idx) previewHistogramIndex = null
  }

  function priorPromptList(side) {
    const summary = session?.w0Summary || {}
    const raw = side === 'neg'
      ? (summary?.neg_prompts || session?.negPromptEnsemble || session?.axis?.neg_prompt_ensemble || session?.axis?.negPromptEnsemble)
      : (summary?.pos_prompts || session?.posPromptEnsemble || session?.axis?.pos_prompt_ensemble || session?.axis?.posPromptEnsemble)
    if (!Array.isArray(raw)) return []
    return raw.map((v) => String(v || '').trim()).filter((v) => v.length > 0)
  }

  function togglePriorPromptMenu(side) {
    if (openPriorPromptSide === side) {
      closePriorPromptMenu()
      return
    }
    editingPriorPromptSide = null
    promptDraftText = ''
    openPriorPromptSide = side
  }

  function closePriorPromptMenu() {
    openPriorPromptSide = null
    editingPriorPromptSide = null
    promptDraftText = ''
  }

  function priorPopupTitle(side) {
    return side === 'neg' ? 'Negative Anchor Text' : 'Positive Anchor Text'
  }

  function startPromptEdit(side) {
    if (promptUpdateBusy) return
    editingPriorPromptSide = side
    promptDraftText = priorPromptList(side).join('\n')
  }

  function cancelPromptEdit() {
    editingPriorPromptSide = null
    promptDraftText = ''
  }

  function parsePromptDraftLines(value) {
    const out = []
    const seen = new Set()
    const parts = String(value || '').split(/\r?\n|\\n/g)
    for (const raw of parts) {
      const text = String(raw || '').trim()
      if (!text) continue
      const key = text.toLowerCase()
      if (seen.has(key)) continue
      seen.add(key)
      out.push(text)
    }
    return out
  }

  function savePromptEdit() {
    const axisId = String(session?.axis?.id || session?.axisId || '').trim()
    if (!axisId || !editingPriorPromptSide) return
    const editedPrompts = parsePromptDraftLines(promptDraftText)
    if (editedPrompts.length === 0) return
    const posPrompts = editingPriorPromptSide === 'pos' ? editedPrompts : posPriorPrompts
    const negPrompts = editingPriorPromptSide === 'neg' ? editedPrompts : negPriorPrompts
    dispatch('updatePrompts', {
      axisId,
      posPrompts,
      negPrompts,
      editedSide: editingPriorPromptSide,
      editedPrompts,
      onSuccess: () => {
        closePriorPromptMenu()
      },
      onError: () => {},
    })
  }

  function openImage(imageId) {
    const id = String(imageId || '').trim()
    if (!id) return
    dispatch('openImage', { imageId: id })
  }

  onMount(() => {
    window.addEventListener('keydown', onWindowFeedbackKeyDown)
    window.addEventListener('keyup', onWindowFeedbackKeyUp)
    window.addEventListener('blur', onWindowFeedbackBlur)
    return () => {
      window.removeEventListener('keydown', onWindowFeedbackKeyDown)
      window.removeEventListener('keyup', onWindowFeedbackKeyUp)
      window.removeEventListener('blur', onWindowFeedbackBlur)
      clearFeedbackDrag()
    }
  })

  onDestroy(() => {
    stopSlicePointerListeners()
    clearFeedbackDrag()
  })

  $: baseScoreEntries = (() => {
    const ids = Array.isArray(session?.ids) ? session.ids : []
    const scores = Array.isArray(session?.scores) ? session.scores : []
    const rawValues = Array.isArray(session?.projectionValues) ? session.projectionValues : []
    const std = Array.isArray(session?.std) ? session.std : []
    const out = []
    for (let i = 0; i < Math.min(ids.length, scores.length); i += 1) {
      out.push({
        id: String(ids[i] || ''),
        score_0_100: Number(scores[i] || 0),
        rawValue: Number(rawValues[i] || 0),
        std: Number(std[i] || 0),
      })
    }
    return out
  })()
  $: scoreEntries = subsetOnly
    ? baseScoreEntries.filter((entry) => subsetIdSet.has(entry.id))
    : baseScoreEntries

  $: undefinedIds = new Set(Array.isArray(session?.undefinedIds) ? session.undefinedIds.map((v) => String(v || '').trim()).filter(Boolean) : [])
  $: rawEntries = scoreEntries.filter((entry) => !undefinedIds.has(entry.id))
  $: rawValuesSorted = rawEntries.map((entry) => Number(entry.rawValue || 0)).filter((value) => Number.isFinite(value)).sort((a, b) => a - b)
  $: rawMin = rawValuesSorted.length > 0 ? rawValuesSorted[0] : 0
  $: rawMax = rawValuesSorted.length > 0 ? rawValuesSorted[rawValuesSorted.length - 1] : 1
  $: rawMid = rawMin + ((rawMax - rawMin) * 0.5)
  $: definedStdValues = rawEntries.map((entry) => Number(entry.std || 0)).filter((value) => Number.isFinite(value))
  $: densityPlot = buildDensityPlot(rawEntries.map((entry) => Number(entry.rawValue || 0)))
  $: histogramBins = buildHistogramBins(rawEntries)
  $: uncertaintyBinAverages = histogramBins.map((bin) => {
    const stdValues = (Array.isArray(bin?.entries) ? bin.entries : [])
      .map((entry) => Number(entry?.std || 0))
      .filter((value) => Number.isFinite(value))
    if (stdValues.length === 0) {
      return { index: Number(bin?.index || 0), avgStd: null }
    }
    return {
      index: Number(bin?.index || 0),
      avgStd: stdValues.reduce((sum, value) => sum + value, 0) / stdValues.length,
    }
  })
  $: finiteUncertaintyBinValues = uncertaintyBinAverages.map((bin) => bin.avgStd).filter((value) => Number.isFinite(value))
  $: uncertaintyBinMin = finiteUncertaintyBinValues.length > 0 ? Math.min(...finiteUncertaintyBinValues) : 0
  $: uncertaintyBinMax = finiteUncertaintyBinValues.length > 0 ? Math.max(...finiteUncertaintyBinValues) : 1
  $: uncertaintyHeatBins = uncertaintyBinAverages.map((bin) => {
    const avgStd = Number(bin?.avgStd)
    const t = Number.isFinite(avgStd) && Math.abs(uncertaintyBinMax - uncertaintyBinMin) > 1e-8
      ? clamp01((avgStd - uncertaintyBinMin) / (uncertaintyBinMax - uncertaintyBinMin))
      : 0.5
    const r = Math.round((34 * (1 - t)) + (239 * t))
    const g = Math.round((197 * (1 - t)) + (68 * t))
    const b = Math.round((94 * (1 - t)) + (68 * t))
    return {
      index: Number(bin?.index || 0),
      avgStd: Number.isFinite(avgStd) ? avgStd : null,
      color: Number.isFinite(avgStd) ? `rgb(${r},${g},${b})` : '#e5e7eb',
    }
  })
  $: uncertaintyTrend = buildUncertaintyTrend(
    rawEntries.map((entry) => Number(entry.rawValue || 0)),
    definedStdValues,
    densityPlot.width,
    densityPlot.height,
  )
  $: exampleDeciles = decileColumns(session?.decileExemplars, subsetOnly ? subsetIdSet : null)
  $: decileMarkers = rawValuesSorted.length === 0 ? [] : exampleDeciles.map((decile) => {
    const q = (decile.index + 0.5) / 10
    const rawValue = quantileSorted(rawValuesSorted, q)
    const xPct = xPctForRawValue(rawValue)
    const y = yForXPct(densityPlot.points, xPct, densityPlot.baseY)
    return {
      index: decile.index,
      rawValue,
      xPct,
      x: (xPct / 100) * densityPlot.width,
      y,
    }
  })
  $: scoreEntryById = new Map(scoreEntries.map((entry) => [entry.id, entry]))
  $: exampleSlots = Array.from({ length: 10 }, (_, idx) => {
    const bin = histogramBins[idx]
    const samples = Array.isArray(bin?.samples) ? bin.samples : []
    return samples.length > 0 ? samples[0] : null
  })
  $: movedIds = new Set(Array.isArray(session?.moves) ? session.moves.map((move) => String(move?.image_id || '')) : [])
  $: movedFeedbackEntries = (() => {
    const ordered = new Map()
    for (const move of Array.isArray(session?.moves) ? session.moves : []) {
      const id = String(move?.image_id || '').trim()
      if (!id || ordered.has(id)) continue
      const entry = scoreEntryById.get(id)
      if (!entry) continue
      ordered.set(id, entry)
    }
    return Array.from(ordered.values()).sort((a, b) => {
      const scoreDelta = Number(a?.score_0_100 || 0) - Number(b?.score_0_100 || 0)
      if (Math.abs(scoreDelta) > 1e-8) return scoreDelta
      return String(a?.id || '').localeCompare(String(b?.id || ''))
    })
  })()
  $: undefinedEntries = Array.from(undefinedIds)
    .map((id) => {
      const entry = scoreEntryById.get(id)
      if (!entry) return null
      return {
        id,
        score_0_100: Number(entry?.score_0_100 || 0),
        rawValue: Number(entry?.rawValue || 0),
        std: Number(entry?.std || 0),
      }
    })
    .filter(Boolean)
  $: if (undefinedEntries.length === 0) {
    undefinedTooltipOpen = false
  }
  $: previewDecile = Number.isInteger(previewDecileIndex) ? (exampleDeciles[previewDecileIndex] || null) : null
  $: previewMarker = Number.isInteger(previewDecileIndex) ? (decileMarkers.find((marker) => marker.index === previewDecileIndex) || null) : null
  $: previewHistogramBin = Number.isInteger(previewHistogramIndex) ? (histogramBins[previewHistogramIndex] || null) : null
  $: negPriorPrompts = priorPromptList('neg')
  $: posPriorPrompts = priorPromptList('pos')
  $: activePriorPrompts = openPriorPromptSide === 'neg' ? negPriorPrompts : (openPriorPromptSide === 'pos' ? posPriorPrompts : [])
  $: {
    const axisId = session?.axis?.id || session?.axisId
    const externalSlice = activeSlice && activeSlice?.axisId === axisId ? activeSlice : null
    if (externalSlice) {
      localActiveSlice = externalSlice
      if (!pendingCommittedSlice || sliceSignature(pendingCommittedSlice) === sliceSignature(externalSlice)) {
        pendingCommittedSlice = null
      }
    } else if (!dragSlice && pendingCommittedSlice && pendingCommittedSlice?.axisId === axisId) {
      localActiveSlice = pendingCommittedSlice
    } else if (!dragSlice) {
      localActiveSlice = null
    }
  }
</script>

<article class="axis-builder-card">
  <div class="axis-builder-header">
    <div class="axis-builder-title-wrap">
      <div class="axis-builder-title">{session?.axis?.name || session?.q || 'Axis'}</div>
    </div>
    <div class="axis-builder-actions">
      <button
        type="button"
        class={`axis-action ${selectedX === session?.axis?.id ? 'active x' : ''}`}
        on:click={() => dispatch('useX', { axis: session?.axis })}
        aria-label="Use axis as X"
        title="Use as X"
      >X</button>
      <button
        type="button"
        class={`axis-action ${selectedY === session?.axis?.id ? 'active y' : ''}`}
        on:click={() => dispatch('useY', { axis: session?.axis })}
        aria-label="Use axis as Y"
        title="Use as Y"
      >Y</button>
      {#if subsetActive}
        <button
          type="button"
          class={`axis-action axis-action-subset ${subsetOnly ? 'active' : ''}`}
          on:click={() => { subsetOnly = !subsetOnly }}
          aria-pressed={subsetOnly}
          title={subsetOnly ? `Showing only subset (${normalizedSubsetIds.length})` : `Filter builder to subset (${normalizedSubsetIds.length})`}
          aria-label={subsetOnly ? `Showing only subset (${normalizedSubsetIds.length})` : `Filter builder to subset (${normalizedSubsetIds.length})`}
        ><MaterialIcon name="replay" /></button>
      {/if}
      <button
        type="button"
        class="axis-action axis-save"
        on:click={() => dispatch('save', {
          axisId: session?.axis?.id || session?.axisId,
          axis: session?.axis,
          q: session?.q || session?.axis?.name || '',
        })}
        aria-label="Save axis"
        title="Save axis to library"
      ><MaterialIcon name="save" /></button>
      <button
        type="button"
        class="axis-action axis-remove"
        on:click={() => dispatch('remove', {
          axisId: session?.axis?.id || session?.axisId,
          axisName: session?.axis?.name || session?.q || session?.axisId || 'Axis',
        })}
        aria-label="Remove axis"
        title="Remove axis"
      ><MaterialIcon name="close" /></button>
    </div>
  </div>

  <div class="distribution-grid">
    <div class="distribution-anchor-cell">
      <button
        type="button"
        class={`prior-end-btn prior-end-btn-neg ${openPriorPromptSide === 'neg' ? 'active' : ''}`}
        aria-label="Inspect negative prior prompts"
        title="Inspect negative prior prompts"
        disabled={negPriorPrompts.length === 0}
        on:pointerdown|stopPropagation={() => {}}
        on:click|stopPropagation={() => togglePriorPromptMenu('neg')}
      ><MaterialIcon name="arrow_circle_left" /></button>
    </div>
    <div class="distribution-shell">
      <div class="histogram-shell">
        {#if localActiveSlice}
          <div
            class="density-slice-overlay"
            style={`left:${clamp100(localActiveSlice.rangeStartPct || 0)}%;width:${Math.max(0.8, clamp100(localActiveSlice.rangeEndPct || 0) - clamp100(localActiveSlice.rangeStartPct || 0))}%;`}
          >
            <button
              type="button"
              class="density-slice-handle density-slice-handle-start"
              aria-label="Adjust slice start"
              title="Adjust slice start"
              on:pointerdown|stopPropagation={($event) => startSliceResize($event, 'start')}
            />
            <button
              type="button"
              class="density-slice-handle density-slice-handle-end"
              aria-label="Adjust slice end"
              title="Adjust slice end"
              on:pointerdown|stopPropagation={($event) => startSliceResize($event, 'end')}
            />
          </div>
        {/if}
        <svg
          class="density-plot histogram-plot"
          viewBox={`0 0 ${HISTOGRAM_WIDTH} ${HISTOGRAM_HEIGHT}`}
          preserveAspectRatio="none"
          role="presentation"
          on:pointerdown|stopPropagation={onPlotPointerDown}
          on:dblclick|stopPropagation={clearSlice}
        >
          {#each histogramBins as bin (bin.index)}
            <rect
              class="histogram-bar"
              x={bin.x}
              y={bin.y}
              width={bin.width}
              height={bin.height}
              rx="0"
              ry="0"
              style={`fill:rgba(37,99,235,${(0.18 + (bin.index * 0.065)).toFixed(3)});`}
            />
          {/each}
        </svg>
        <div
          class="histogram-hit-layer"
          role="presentation"
          aria-hidden="true"
          on:pointerdown|stopPropagation={onPlotPointerDown}
          on:dblclick|stopPropagation={clearSlice}
        >
          {#each histogramBins as bin (bin.index)}
            <button
              type="button"
              class={`histogram-hit ${previewHistogramIndex === bin.index ? 'active' : ''}`}
              style={`left:${bin.centerPct}%;width:${(bin.width / HISTOGRAM_WIDTH) * 100}%;`}
              aria-label={`Histogram bin ${bin.index + 1}`}
              on:mouseenter={() => onHistogramEnter(bin.index)}
              on:mouseleave={() => onHistogramLeave(bin.index)}
              on:focus={() => onHistogramEnter(bin.index)}
              on:blur={() => onHistogramLeave(bin.index)}
            />
          {/each}
        </div>
        {#if previewHistogramBin && Array.isArray(previewHistogramBin.samples) && previewHistogramBin.samples.length > 0}
          <div
            class="decile-preview-pop histogram-preview-pop"
            aria-hidden="true"
            style={`left:${previewHistogramBin.centerPct}%;top:${Math.max(18, (previewHistogramBin.y / Math.max(1, HISTOGRAM_HEIGHT)) * 100)}%;`}
          >
            {#each previewHistogramBin.samples as sample (sample.id)}
              {@const sampleItem = resolveItem(sample.id)}
              <div class="decile-preview-thumb">
                {#if sampleItem?.thumbUrl || sampleItem?.url}
                  <img src={sampleItem.thumbUrl || sampleItem.url} alt="" class="decile-preview-thumb-img" />
                {:else}
                  <div class="decile-preview-thumb-img decile-thumb-empty" aria-hidden="true" />
                {/if}
              </div>
            {/each}
          </div>
        {/if}
        {#if openPriorPromptSide}
          <div
            class={`prior-popup ${openPriorPromptSide === 'neg' ? 'left' : 'right'}`}
            role="dialog"
            aria-label={priorPopupTitle(openPriorPromptSide)}
            on:pointerdown|stopPropagation
          >
            <div class="prior-popup-head">
              <span>{priorPopupTitle(openPriorPromptSide)}</span>
              <div class="prior-popup-head-actions">
                {#if editingPriorPromptSide === openPriorPromptSide}
                  <button
                    type="button"
                    class="prior-popup-head-btn"
                    aria-label="Save anchor prompts"
                    title="Save anchor prompts"
                    disabled={promptUpdateBusy}
                    on:click|stopPropagation={savePromptEdit}
                  ><MaterialIcon name="save" /></button>
                  <button
                    type="button"
                    class="prior-popup-head-btn"
                    aria-label="Cancel prompt editing"
                    title="Cancel prompt editing"
                    disabled={promptUpdateBusy}
                    on:click|stopPropagation={cancelPromptEdit}
                  ><MaterialIcon name="close" /></button>
                {:else}
                  <button
                    type="button"
                    class="prior-popup-head-btn"
                    aria-label="Edit anchor prompts"
                    title="Edit anchor prompts"
                    disabled={promptUpdateBusy}
                    on:click|stopPropagation={() => startPromptEdit(openPriorPromptSide)}
                  ><MaterialIcon name="edit" /></button>
                  <button
                    type="button"
                    class="prior-popup-close"
                    aria-label="Close anchor prompt list"
                    on:click|stopPropagation={closePriorPromptMenu}
                  ><MaterialIcon name="close" /></button>
                {/if}
              </div>
            </div>
            {#if editingPriorPromptSide === openPriorPromptSide}
              <div class="prior-popup-editor">
                <textarea
                  class="prior-popup-textarea"
                  rows="8"
                  bind:value={promptDraftText}
                  placeholder="One prompt per line"
                  disabled={promptUpdateBusy}
                />
              </div>
            {:else if activePriorPrompts.length > 0}
              <div class="prior-popup-list">
                {#each activePriorPrompts as prompt, idx (`${openPriorPromptSide}-${idx}-${prompt}`)}
                  <div class="prior-popup-row">
                    <span class="prior-popup-index">{idx + 1}.</span>
                    <span class="prior-popup-text">{prompt}</span>
                  </div>
                {/each}
              </div>
            {:else}
              <div class="prior-popup-empty">No prompts available.</div>
            {/if}
          </div>
        {/if}
      </div>
    </div>
    <div class="distribution-anchor-cell">
      <button
        type="button"
        class={`prior-end-btn prior-end-btn-pos ${openPriorPromptSide === 'pos' ? 'active' : ''}`}
        aria-label="Inspect positive prior prompts"
        title="Inspect positive prior prompts"
        disabled={posPriorPrompts.length === 0}
        on:pointerdown|stopPropagation={() => {}}
        on:click|stopPropagation={() => togglePriorPromptMenu('pos')}
      ><MaterialIcon name="arrow_circle_right" /></button>
    </div>
  </div>

  <div class="decile-ribbons">
    <div class="scale-row uncertainty-strip-row" aria-label="Average uncertainty by rating bin">
      <span class="decile-row-label uncertainty-strip-label" aria-label="Uncertainty" title="Uncertainty">Uncertainty</span>
      <div class="decile-track uncertainty-track">
        {#each uncertaintyHeatBins as bin (bin.index)}
          <span
            class="uncertainty-strip-cell"
            style={`background:${bin.color};`}
            title={bin.avgStd === null ? `Rating ${bin.index + 1}: no samples` : `Rating ${bin.index + 1}: average uncertainty ${formatRawValue(bin.avgStd)}`}
          />
        {/each}
      </div>
      <span class="scale-undefined-spacer" aria-hidden="true" />
    </div>

    <div class="scale-row decile-row-wrap">
      <span class="decile-row-label" aria-label="Examples" title="Examples">Examples</span>
      <div class="decile-track decile-row" role="list" aria-label="Representative examples by histogram bin">
        {#each exampleSlots as exemplar, idx (`example-${idx}-${exemplar?.id || 'empty'}`)}
          {@const item = exemplar ? resolveItem(exemplar.id) : null}
          <div
            role="listitem"
            class={`decile-slot ${dropDecileIndex === idx ? 'drop-active' : ''}`}
            on:mouseenter={() => onDecileEnter(idx)}
            on:mouseleave={() => onDecileLeave(idx)}
            on:dragover={allowDrop}
            on:dragenter={() => onDropZoneEnter(idx)}
            on:dragleave={() => onDropZoneLeave(idx)}
            on:drop={(e) => onDropZone(e, idx)}
          >
            <button
              type="button"
              class={`decile-thumb-btn ${exemplar && movedIds.has(exemplar.id) ? 'moved' : ''}`}
              draggable={!busy && !!exemplar}
              on:dragstart={(e) => { if (exemplar) onDragStart(e, exemplar, { sourceRow: 'examples', sourceRank: idx + 1 }) }}
              on:click|stopPropagation={() => { if (exemplar?.id) openImage(exemplar.id) }}
              title={exemplar?.id || `Example ${idx + 1}`}
            >
              {#if item?.thumbUrl || item?.url}
                <img src={item.thumbUrl || item.url} alt={exemplar?.id || `Example ${idx + 1}`} class="decile-thumb" />
              {:else}
                <div class="decile-thumb decile-thumb-empty" aria-hidden="true" />
              {/if}
            </button>
          </div>
        {/each}
      </div>
      <div class="undefined-slot-wrap">
        <button
          type="button"
          class={`undefined-slot-btn ${dropUndefinedActive ? 'drop-active' : ''}`}
          aria-label="Undefined images"
          title="Undefined images"
          on:click|stopPropagation={toggleUndefinedTooltip}
          on:dragover={allowDrop}
          on:dragenter={onUndefinedDropEnter}
          on:dragleave={onUndefinedDropLeave}
          on:drop={onUndefinedDrop}
        ><MaterialIcon name="close" size={22} /></button>
        {#if undefinedTooltipOpen}
          <div class="undefined-tooltip" role="dialog" aria-label="Undefined images" on:pointerdown|stopPropagation>
            <div class="undefined-tooltip-head">
              <span>Undefined</span>
              <button
                type="button"
                class="undefined-tooltip-close"
                aria-label="Close undefined images"
                title="Close"
                on:click|stopPropagation={closeUndefinedTooltip}
              ><MaterialIcon name="close" size={16} /></button>
            </div>
            {#if undefinedEntries.length > 0}
              <div class="undefined-tooltip-grid">
                {#each undefinedEntries as entry (entry.id)}
                  {@const item = resolveItem(entry.id)}
                  <button
                    type="button"
                    class="undefined-thumb-btn"
                    draggable={!busy}
                    on:dragstart={(e) => onDragStart(e, entry, { sourceRow: 'undefined', sourceRank: 'undefined' })}
                    on:click|stopPropagation={() => openImage(entry.id)}
                    title={entry.id}
                  >
                    {#if item?.thumbUrl || item?.url}
                      <img src={item.thumbUrl || item.url} alt={entry.id} class="undefined-thumb" />
                    {:else}
                      <div class="undefined-thumb decile-thumb-empty" aria-hidden="true" />
                    {/if}
                  </button>
                {/each}
              </div>
            {:else}
              <div class="undefined-tooltip-empty">No undefined images</div>
            {/if}
          </div>
        {/if}
      </div>
    </div>

    <div class="scale-row decile-row-wrap">
      <span class="decile-row-label" aria-label="Uncertain" title="Uncertain">Uncertain</span>
      <div class="decile-track decile-row" role="list" aria-label="Most uncertain examples by decile">
        {#each exampleDeciles as decile (decile.index)}
          {@const exemplar = decile.leastCertain}
          {@const item = exemplar ? resolveItem(exemplar.id) : null}
          <div
            role="listitem"
            class={`decile-slot ${dropDecileIndex === decile.index ? 'drop-active' : ''}`}
            on:mouseenter={() => onDecileEnter(decile.index)}
            on:mouseleave={() => onDecileLeave(decile.index)}
            on:dragover={allowDrop}
            on:dragenter={() => onDropZoneEnter(decile.index)}
            on:dragleave={() => onDropZoneLeave(decile.index)}
            on:drop={(e) => onDropZone(e, decile.index)}
          >
            <button
              type="button"
              class={`decile-thumb-btn ${exemplar && movedIds.has(exemplar.id) ? 'moved' : ''}`}
              draggable={!busy && !!exemplar}
              on:dragstart={(e) => { if (exemplar) onDragStart(e, exemplar, { sourceRow: 'least sure', sourceRank: decile.index + 1 }) }}
              on:click|stopPropagation={() => { if (exemplar?.id) openImage(exemplar.id) }}
              title={exemplar?.id || decile.label}
            >
              {#if item?.thumbUrl || item?.url}
                <img src={item.thumbUrl || item.url} alt={exemplar?.id || decile.label} class="decile-thumb" />
              {:else}
                <div class="decile-thumb decile-thumb-empty" aria-hidden="true" />
              {/if}
            </button>
          </div>
        {/each}
      </div>
      <span class="scale-undefined-spacer" aria-hidden="true" />
    </div>

    <div class="scale-row decile-row-wrap feedback-row-wrap">
      <span class="decile-row-label feedback-row-label" aria-label="Refinement summary" title="Refinement summary">Refinement
summary</span>
      <div
        class={`feedback-track ${feedbackDrag ? 'dragging' : ''}`}
        aria-label="Moved image positions"
        title="Drag dots to retarget moved images. Hold Delete and click a dot to remove that move."
      >
        <div class="feedback-line" />
        <span class="feedback-end feedback-end-left">0</span>
        <span class="feedback-end feedback-end-right">1</span>
        {#each movedFeedbackEntries as entry (entry.id)}
          {@const item = resolveItem(entry.id)}
          <button
            type="button"
            class={`feedback-point ${feedbackDrag?.imageId === entry.id ? 'dragging' : ''} ${deleteModifierActive ? 'delete-armed' : ''}`}
            style={`left:${feedbackDisplayPct(entry)}%;`}
            aria-label={`Moved image ${entry.id}`}
            title={entry.id}
            on:mouseenter={() => { hoveredFeedbackId = entry.id }}
            on:mouseleave={() => { if (hoveredFeedbackId === entry.id) hoveredFeedbackId = '' }}
            on:focus={() => { hoveredFeedbackId = entry.id }}
            on:blur={() => { if (hoveredFeedbackId === entry.id) hoveredFeedbackId = '' }}
            on:pointerdown|stopPropagation={(e) => startFeedbackPointInteraction(e, entry)}
          />
          {#if hoveredFeedbackId === entry.id}
            <div class="feedback-hover" style={`left:${feedbackDisplayPct(entry)}%;`}>
              {#if item?.thumbUrl || item?.url}
                <img src={item.thumbUrl || item.url} alt={entry.id} class="feedback-hover-image" />
              {:else}
                <div class="feedback-hover-image decile-thumb-empty" aria-hidden="true" />
              {/if}
            </div>
          {/if}
        {/each}
      </div>
      <span class="scale-undefined-spacer" aria-hidden="true" />
    </div>
  </div>
</article>

<style>
  .axis-builder-card,
  .decile-ribbons {
    --axis-bin-size: 50px;
    --axis-label-width: 64px;
    --axis-undefined-width: 50px;
  }

  .axis-builder-card {
    border: 1px solid #d9d9dd;
    border-radius: 8px;
    background: #ffffff;
    padding: 8px 10px 9px;
    display: grid;
    gap: 6px;
    overflow: visible;
  }

  .axis-builder-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
  }

  .axis-builder-title {
    min-width: 0;
    font-size: var(--font-size-body);
    font-weight: 700;
    line-height: 1.1;
    color: #3d4257;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .axis-builder-title-wrap {
    min-width: 0;
    display: flex;
    align-items: center;
  }

  .axis-builder-actions {
    display: inline-flex;
    align-items: center;
    gap: 2px;
    flex: none;
  }

  .axis-action {
    width: 28px;
    height: 28px;
    border: 1px solid #d5d8e7;
    border-radius: 6px;
    background: #ffffff;
    color: #64748b;
    font-size: var(--font-size-body);
    line-height: 1;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    transition: border-color 120ms ease, color 120ms ease, background-color 120ms ease;
  }

  .axis-action.active {
    border-color: #60a5fa;
    background: #eff6ff;
    color: #1d4ed8;
  }

  .axis-action.axis-remove {
    color: #dc2626;
    border-color: #fecaca;
    background: #fff5f5;
  }

  .axis-action.axis-save {
    color: #64748b;
    border-color: #d5d8e7;
    background: #ffffff;
  }

  .density-shell {
    position: relative;
    border-top: 1px solid #ececf2;
    padding-top: 8px;
    overflow: visible;
  }

  .distribution-grid,
  .builder-density-axis,
  .scale-row {
    display: grid;
    grid-template-columns: var(--axis-label-width) minmax(0, 1fr) var(--axis-undefined-width);
    align-items: center;
    column-gap: 0;
  }

  .distribution-shell {
    position: relative;
    border-top: 1px solid #ececf2;
    padding-top: 4px;
    overflow: visible;
  }

  .scale-side-spacer,
  .scale-undefined-spacer,
  .undefined-track-spacer {
    min-width: 0;
    min-height: 1px;
  }

  .distribution-anchor-cell {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 61px;
  }

  .density-slice-overlay {
    position: absolute;
    top: 0;
    bottom: 0;
    background: rgba(96, 165, 250, 0.16);
    border-left: 1px solid rgba(96, 165, 250, 0.38);
    border-right: 1px solid rgba(96, 165, 250, 0.38);
    pointer-events: none;
    z-index: 1;
  }

  .density-slice-handle {
    position: absolute;
    top: 50%;
    width: 12px;
    height: 24px;
    transform: translate(-50%, -50%);
    border: 1px solid rgba(59, 130, 246, 0.58);
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.96);
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.2);
    pointer-events: auto;
    cursor: ew-resize;
  }

  .density-slice-handle-start {
    left: 0;
  }

  .density-slice-handle-end {
    left: 100%;
  }

  .density-plot {
    position: relative;
    z-index: 2;
    display: block;
    width: 100%;
    height: 61px;
    margin-top: 0;
    user-select: none;
    cursor: col-resize;
  }

  .density-line {
    fill: none;
    stroke: #2563eb;
    stroke-width: 2.3;
    stroke-linecap: round;
    stroke-linejoin: round;
  }

  .uncert-band {
    fill: rgba(100, 116, 139, 0.08);
    stroke: none;
  }

  .uncert-line {
    fill: none;
    stroke: #94a3b8;
    stroke-width: 1.8;
    stroke-linecap: round;
    stroke-linejoin: round;
    stroke-dasharray: 4 3;
  }

  .density-legend {
    position: absolute;
    bottom: 6px;
    right: 10px;
    z-index: 4;
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 4px 6px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.9);
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
    color: #636a7f;
    font-size: var(--font-size-small);
    line-height: 1;
  }

  .density-legend-item {
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }

  .density-legend-line {
    width: 18px;
    height: 0;
    border-top: 2px solid currentColor;
    display: inline-block;
  }

  .density-legend-line-dist {
    color: #2563eb;
  }

  .density-legend-line-uncert {
    color: #94a3b8;
    border-top-style: dashed;
  }

  .density-dot-layer {
    position: absolute;
    inset: 6px 0 0;
    pointer-events: none;
    z-index: 3;
  }

  .density-dot {
    position: absolute;
    width: 60px;
    height: 60px;
    border: 0;
    padding: 0;
    margin: 0;
    display: block;
    appearance: none;
    border-radius: 999px;
    background: transparent;
    color: #2563eb;
    cursor: pointer;
    pointer-events: auto;
    transition: transform 120ms ease;
    transform: translate(-50%, -50%);
    box-shadow: none;
  }

  .density-dot::before {
    content: '';
    position: absolute;
    left: 50%;
    top: 50%;
    width: 8px;
    height: 8px;
    border-radius: 999px;
    background: currentColor;
    box-shadow: 0 0 0 1px currentColor;
    transform: translate(-50%, -50%);
    transition: transform 120ms ease, background-color 120ms ease, box-shadow 120ms ease;
  }

  .density-dot.active {
    color: #1d4ed8;
  }

  .density-dot.active::before {
    transform: translate(-50%, -50%) scale(1.08);
  }

  .decile-preview-pop {
    position: absolute;
    z-index: 120;
    transform: translateX(-50%);
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid #d8dbe7;
    box-shadow: 0 12px 24px rgba(15, 23, 42, 0.12);
    pointer-events: none;
  }

  .histogram-preview-pop {
    transform: translate(-50%, -100%);
  }

  .histogram-shell {
    position: relative;
    min-height: 61px;
    overflow: visible;
  }

  .histogram-plot {
    cursor: col-resize;
  }

  .histogram-bar {
    stroke-width: 0;
  }

  .histogram-hit-layer {
    position: absolute;
    inset: 0;
    z-index: 3;
    pointer-events: none;
  }

  .histogram-hit {
    position: absolute;
    top: 0;
    bottom: 0;
    transform: translateX(-50%);
    border: 0;
    padding: 0;
    margin: 0;
    background: transparent;
    pointer-events: auto;
  }

  .decile-preview-thumb {
    width: 84px;
    height: 84px;
    border-radius: 6px;
    overflow: hidden;
    flex: none;
    background: #f4f4f6;
  }

  .decile-preview-thumb-img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }

  .prior-end-btn {
    position: relative;
    top: auto;
    transform: none;
    width: 36px;
    height: 36px;
    border: 1px solid rgba(191, 219, 254, 0.9);
    border-radius: 8px;
    background: rgba(248, 250, 252, 0.94);
    color: #2563eb;
    font-size: 12px;
    font-weight: 700;
    line-height: 1;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    z-index: 4;
    box-shadow: none;
  }

  .prior-end-btn:disabled {
    opacity: 0.45;
    color: #8b8fa8;
  }

  .prior-end-btn.active {
    border-color: #60a5fa;
    background: #eff6ff;
  }

  .prior-end-btn-neg {
    left: auto;
  }

  .prior-end-btn-pos {
    right: auto;
  }

  .prior-popup {
    position: absolute;
    top: 10px;
    width: min(520px, calc(100% - 8px));
    max-height: 320px;
    border: 1px solid #d8dbe7;
    border-radius: 9px;
    background: #ffffff;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.16);
    z-index: 6;
    display: grid;
    grid-template-rows: auto 1fr;
    overflow: hidden;
  }

  .prior-popup.left {
    left: 0;
  }

  .prior-popup.right {
    right: 0;
  }

  .prior-popup-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 6px 8px;
    border-bottom: 1px solid #eceef6;
    font-size: var(--font-size-small);
    font-weight: 600;
    color: #4a4f67;
  }

  .prior-popup-head-actions {
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }

  .prior-popup-head-btn,
  .prior-popup-close {
    border: 0;
    background: transparent;
    color: #64748b;
    font-size: 14px;
    line-height: 1;
    padding: 0;
    width: 18px;
    height: 18px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }

  .prior-popup-head-btn:disabled,
  .prior-popup-close:disabled {
    opacity: 0.45;
  }

  .prior-popup-list {
    overflow: auto;
    padding: 6px 8px 7px;
    display: grid;
    gap: 5px;
  }

  .prior-popup-row {
    display: grid;
    grid-template-columns: 16px 1fr;
    gap: 6px;
    align-items: start;
    font-size: var(--font-size-small);
    color: #545a71;
    line-height: 1.28;
  }

  .prior-popup-index {
    color: #8a8ea3;
  }

  .prior-popup-text {
    word-break: break-word;
  }

  .prior-popup-empty {
    padding: 8px;
    font-size: var(--font-size-small);
    color: #83889d;
  }

  .prior-popup-editor {
    padding: 8px;
    min-height: 0;
    display: flex;
  }

  .prior-popup-textarea {
    width: 100%;
    min-height: 220px;
    margin: 0;
    resize: vertical;
    line-height: 1.35;
  }

  .decile-ribbons {
    display: grid;
    gap: 3px;
    padding: 1px 0 6px;
    border-radius: 0;
    background: transparent;
    box-shadow: none;
  }

  .decile-track {
    display: grid;
    grid-template-columns: repeat(10, minmax(0, 1fr));
    gap: 0;
    align-items: center;
    min-width: 0;
  }

  .uncertainty-strip-row {
    min-height: 22px;
    margin-top: -1px;
    margin-bottom: 1px;
  }

  .uncertainty-strip-label {
    min-height: 22px;
  }

  .uncertainty-track {
    height: 10px;
    overflow: hidden;
    border-radius: 3px;
  }

  .uncertainty-strip-cell {
    display: block;
    width: 100%;
    height: 10px;
  }

  .decile-row {
    min-width: 0;
  }

  .decile-row-wrap {
    min-width: 0;
  }

  .decile-row-label {
    width: var(--axis-label-width);
    min-height: var(--axis-bin-size);
    display: inline-flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 4px;
    color: #7b8197;
    font-size: 11px;
    font-weight: 700;
    line-height: 1.1;
    white-space: pre-line;
    text-align: right;
    color: #64748b;
  }

  .feedback-row-wrap {
    margin-top: 2px;
  }

  .feedback-row-label {
    min-height: var(--axis-bin-size);
  }

  .feedback-track {
    position: relative;
    height: 22px;
    min-width: 0;
  }

  .feedback-line {
    position: absolute;
    left: 0;
    right: 0;
    top: 10px;
    height: 2px;
    border-radius: 999px;
    background: linear-gradient(90deg, rgba(148, 163, 184, 0.72) 0%, rgba(37, 99, 235, 0.84) 100%);
  }

  .feedback-end {
    position: absolute;
    top: 0;
    font-size: 10px;
    line-height: 1;
    color: #94a3b8;
  }

  .feedback-end-left {
    left: 0;
  }

  .feedback-end-right {
    right: 0;
  }

  .feedback-point {
    position: absolute;
    top: 11px;
    width: 12px;
    height: 12px;
    transform: translate(-50%, -50%);
    border: 2px solid #ffffff;
    border-radius: 999px;
    background: #2563eb;
    box-shadow: 0 4px 10px rgba(37, 99, 235, 0.25);
    padding: 0;
    margin: 0;
    cursor: grab;
    touch-action: none;
    transition: transform 120ms ease, box-shadow 120ms ease, background-color 120ms ease;
  }

  .feedback-point.dragging {
    cursor: grabbing;
    transform: translate(-50%, -50%) scale(1.18);
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.32);
  }

  .feedback-point.delete-armed {
    background: #dc2626;
    box-shadow: 0 4px 10px rgba(220, 38, 38, 0.24);
  }

  .feedback-hover {
    position: absolute;
    bottom: calc(100% + 6px);
    transform: translateX(-50%);
    width: 56px;
    height: 56px;
    padding: 4px;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.96);
    border: 1px solid #d8dbe7;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.14);
    pointer-events: none;
    z-index: 125;
  }

  .feedback-hover-image {
    width: 100%;
    height: 100%;
    border-radius: 6px;
    object-fit: cover;
    display: block;
    background: #f4f4f6;
  }

  .undefined-slot-wrap {
    position: relative;
    width: var(--axis-undefined-width);
    height: var(--axis-bin-size);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-left: 1px;
  }

  .undefined-slot-btn {
    width: var(--axis-bin-size);
    height: var(--axis-bin-size);
    border: 0;
    border-radius: 4px;
    background: #e5e7eb;
    color: #6b7280;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    transition: background-color 120ms ease, color 120ms ease, box-shadow 120ms ease;
  }

  .undefined-slot-btn.drop-active {
    background: #dbeafe;
    color: #2563eb;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.24);
  }

  .undefined-tooltip {
    position: absolute;
    top: calc(100% + 8px);
    right: 0;
    z-index: 130;
    width: min(320px, 60vw);
    padding: 8px;
    border: 1px solid #d8dbe7;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.97);
    box-shadow: 0 14px 32px rgba(15, 23, 42, 0.18);
    display: grid;
    gap: 8px;
  }

  .undefined-tooltip-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    font-size: var(--font-size-small);
    font-weight: 600;
    color: #4a4f67;
  }

  .undefined-tooltip-close {
    width: 20px;
    height: 20px;
    border: 0;
    padding: 0;
    background: transparent;
    color: #64748b;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }

  .undefined-tooltip-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(46px, 1fr));
    gap: 6px;
  }

  .undefined-thumb-btn {
    width: 100%;
    aspect-ratio: 1 / 1;
    border: 0;
    padding: 0;
    border-radius: 4px;
    background: transparent;
    overflow: visible;
  }

  .undefined-thumb {
    width: 100%;
    height: 100%;
    display: block;
    object-fit: cover;
    border-radius: 4px;
    background: #f4f4f6;
  }

  .undefined-tooltip-empty {
    font-size: var(--font-size-small);
    color: #83889d;
  }

  .decile-slot {
    width: 100%;
    aspect-ratio: 1 / 1;
    border-radius: 4px;
    position: relative;
    padding: 1px;
    box-sizing: border-box;
  }

  .representative-slot {
    cursor: default;
  }

  .decile-slot.drop-active {
    outline: 2px solid rgba(37, 99, 235, 0.35);
    outline-offset: 1px;
  }

  .decile-thumb-btn {
    width: 100%;
    height: 100%;
    border: 0;
    border-radius: 4px;
    overflow: visible;
    padding: 0;
    position: relative;
    background: transparent;
    transition: z-index 120ms ease;
  }

  .decile-thumb-btn:hover,
  .decile-thumb-btn:focus-visible {
    z-index: 2;
  }

  .decile-thumb-btn.moved .decile-thumb {
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.42);
  }

  .representative-thumb-btn:disabled {
    cursor: default;
  }

  .decile-thumb {
    width: 100%;
    height: 100%;
    display: block;
    object-fit: cover;
    user-select: none;
    -webkit-user-drag: none;
    border-radius: 4px;
    background: #f4f4f6;
    box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.08);
    transition: transform 120ms ease, box-shadow 120ms ease;
    transform-origin: center center;
  }

  .decile-thumb-btn:hover .decile-thumb,
  .decile-thumb-btn:focus-visible .decile-thumb {
    transform: scale(1.92);
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.24);
  }

  .decile-thumb-empty {
    background: #ececf1;
  }

  @media (max-width: 720px) {
    .axis-builder-card,
    .decile-ribbons {
      --axis-bin-size: 46px;
      --axis-label-width: 58px;
      --axis-undefined-width: 46px;
    }
  }
</style>
