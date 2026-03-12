<script>
  import { createEventDispatcher, onDestroy } from 'svelte'
  import AxisBuilder from './AxisBuilder.svelte'
  import { axisBuildersStore } from '../lib/axisBuilderStore'

  export let axes = []
  export let items = []
  export let selectedX = null
  export let selectedY = null
  export let apiBase = 'http://127.0.0.1:5002'
  export let dataset = ''
  export let externalSlices = []

  const dispatch = createEventDispatcher()

  let promptText = ''
  let extractedPromptText = ''
  let extracting = false
  let creatingAxes = false
  let errorMsg = ''
  let extractionInfo = ''
  let attributeChips = []
  let proposals = []
  let activeSlices = []
  let dragSlice = null
  let dragCalibration = null
  let recommendingScatterplots = false
  let recommendationInfo = ''
  let recommendations = []
  let manualAxisText = ''
  let hoveredExplainChipId = ''
  const HISTOGRAM_MAX_BINS = 6
  $: showLoadingBackdrop = Boolean(extracting || creatingAxes)
  $: loadingBackdropText = extracting ? 'Running LLM...' : 'Creating axes...'

  $: axisById = new Map((axes || []).map((axis) => [axis.id, axis]))
  $: itemById = new Map((items || []).map((item) => [String(item?.id || ''), item]).filter((row) => row[0]))
  $: selectedChips = Array.isArray(attributeChips) ? attributeChips.filter((chip) => chip?.selected) : []
  $: selectedChipCount = selectedChips.length
  $: hoveredExplainChip = Array.isArray(attributeChips)
    ? attributeChips.find((chip) => chip?.id === hoveredExplainChipId) || null
    : null
  $: promptSupportSegments = buildPromptHighlightSegments(
    extractedPromptText || promptText,
    hoveredExplainChip?.supportSpans || [],
  )
  $: axisBuilders = $axisBuildersStore
  $: candidateAxes = Array.isArray(axes)
    ? axes
      .filter((axis) => axis?.id && axis?.coords && Object.keys(axis.coords || {}).length > 1)
      .map((axis) => ({
        id: axis.id,
        name: axis.name || axis.id,
        coords: axis.coords || {},
        attribute_type: axis.attribute_type || axis.type || '',
        scoring_method: axis.scoring_method || '',
        group: axis.group || '',
      }))
    : []
  function normalizedSliceArray(raw) {
    if (!Array.isArray(raw)) return []
    const out = []
    for (const s of raw) {
      if (!s || typeof s !== 'object') continue
      const axisId = String(s.axisId || '').trim()
      if (!axisId) continue
      const ids = Array.isArray(s.ids) ? s.ids.map((id) => String(id || '').trim()).filter(Boolean) : []
      out.push({ ...s, axisId, ids })
    }
    return out
  }

  function sliceSignature(raw) {
    return JSON.stringify(
      normalizedSliceArray(raw).map((slice) => ({
        axisId: slice.axisId,
        rangeStartPct: Number(slice.rangeStartPct ?? 0),
        rangeEndPct: Number(slice.rangeEndPct ?? 0),
        ids: Array.isArray(slice.ids) ? slice.ids : [],
      })),
    )
  }

  $: {
    const nextSig = sliceSignature(externalSlices)
    const prevSig = sliceSignature(activeSlices)
    if (nextSig !== prevSig) {
      activeSlices = normalizedSliceArray(externalSlices)
    }
  }

  function updateActiveSlice(nextSlice) {
    const incoming = nextSlice && typeof nextSlice === 'object' ? nextSlice : null
    const axisId = String(incoming?.axisId || '').trim()
    const prev = normalizedSliceArray(activeSlices)
    const filtered = axisId ? prev.filter((s) => String(s.axisId || '').trim() !== axisId) : prev
    if (incoming && axisId) {
      const ids = Array.isArray(incoming.ids) ? incoming.ids.map((id) => String(id || '').trim()).filter(Boolean) : []
      activeSlices = [...filtered, { ...incoming, axisId, ids }]
    } else {
      activeSlices = filtered
    }
    dispatch('sliceChange', { slices: activeSlices, slice: activeSlices.length === 1 ? activeSlices[0] : null })
  }

  function clearSlice() {
    activeSlices = []
    dragSlice = null
    window.removeEventListener('pointermove', onWindowPointerMove)
    dispatch('sliceChange', { slices: [], slice: null })
  }

  function removeSliceForAxis(axisId) {
    const id = String(axisId || '').trim()
    if (!id) return
    const next = normalizedSliceArray(activeSlices).filter((slice) => String(slice.axisId || '').trim() !== id)
    activeSlices = next
    dispatch('sliceChange', { slices: activeSlices, slice: activeSlices.length === 1 ? activeSlices[0] : null })
  }

  function sliceForAxis(axisId) {
    const id = String(axisId || '').trim()
    if (!id) return null
    const arr = normalizedSliceArray(activeSlices)
    return arr.find((s) => s.axisId === id) || null
  }

  function axisExists(axisId) {
    return axisById.has(axisId)
  }

  function emitAxis(axis) {
    if (!axis || !axis.id) return
    dispatch('upsertAxis', { axis })
  }

  function emitAxisRemoval(axisId) {
    const id = String(axisId || '').trim()
    if (!id) return
    dispatch('removeAxis', { id })
  }

  function applyAxisX(proposal) {
    if (!proposal?.axis?.id) return
    emitAxis(proposal.axis)
    dispatch('setX', { id: proposal.axis.id })
  }

  function applyAxisY(proposal) {
    if (!proposal?.axis?.id) return
    emitAxis(proposal.axis)
    dispatch('setY', { id: proposal.axis.id })
  }

  function addNextAttribute() {
    if (!Array.isArray(proposals) || proposals.length === 0) return
    const next = proposals.find((p) => p?.axis?.id && !axisExists(p.axis.id))
    if (!next) return
    emitAxis(next.axis)
  }

  function onAxisDragStart(e, proposal) {
    if (!proposal?.axis?.id) return
    emitAxis(proposal.axis)
    try {
      e.dataTransfer.setData('application/axis-id', proposal.axis.id)
      e.dataTransfer.setData('text/plain', proposal.axis.id)
      e.dataTransfer.effectAllowed = 'copyMove'
    } catch (_) {}
  }

  function axisScoreEntries(axis) {
    const coords = axis?.coords || {}
    const out = []
    for (const [id, raw] of Object.entries(coords)) {
      const v = Number(raw)
      if (!Number.isFinite(v)) continue
      if (!itemById.has(id)) continue
      out.push({ id, score: clamp01(v) })
    }
    return out
  }

  function pickNearestEntry(entries, target, used = null) {
    if (!Array.isArray(entries) || entries.length === 0) return null
    const t = clamp01(target)
    let best = null
    let bestD = Number.POSITIVE_INFINITY
    for (const e of entries) {
      if (used && used.has(e.id)) continue
      const d = Math.abs(Number(e.score || 0) - t)
      if (d < bestD) {
        bestD = d
        best = e
      }
    }
    if (best) return best
    // Fallback to reuse when not enough unique examples.
    for (const e of entries) {
      const d = Math.abs(Number(e.score || 0) - t)
      if (d < bestD) {
        bestD = d
        best = e
      }
    }
    return best
  }

  function buildContinuousCalibration(proposal) {
    const entries = axisScoreEntries(proposal?.axis)
    const used = new Set()
    const buckets = []
    for (let i = 0; i <= 10; i += 1) {
      const target = i / 10
      const ex = pickNearestEntry(entries, target, used)
      if (ex) used.add(ex.id)
      buckets.push({
        label: `${i * 10}%`,
        target,
        ids: ex ? [ex.id] : [],
      })
    }
    return { kind: 'continuous', buckets }
  }

  function buildCategoricalCalibration(proposal) {
    const values = Array.isArray(proposal?.values) ? proposal.values.map((v) => String(v || '').trim()).filter(Boolean) : []
    if (values.length === 0) return buildContinuousCalibration(proposal)

    const entries = axisScoreEntries(proposal?.axis)
    const predictedById = proposal?.predictedById || {}
    const pools = new Map(values.map((v) => [v.toLowerCase(), []]))
    for (const e of entries) {
      const pred = String(predictedById[e.id] || '').trim().toLowerCase()
      if (pred && pools.has(pred)) pools.get(pred).push(e)
    }

    const used = new Set()
    const buckets = values.map((label, idx) => {
      const target = values.length <= 1 ? 0.5 : (idx / (values.length - 1))
      const local = pools.get(label.toLowerCase()) || []
      let chosen = pickNearestEntry(local, target, used)
      if (!chosen) chosen = pickNearestEntry(entries, target, used)
      if (chosen) used.add(chosen.id)
      return {
        label,
        target,
        ids: chosen ? [chosen.id] : [],
      }
    })
    return { kind: proposal?.variableType || 'categorical', buckets }
  }

  function buildCalibration(proposal) {
    const type = normalizeAttributeType(proposal?.variableType || proposal?.axis?.attribute_type || '')
    if (type === 'continuous') return buildContinuousCalibration(proposal)
    return buildCategoricalCalibration(proposal)
  }

  function updateProposal(proposalKey, updater) {
    let updated = null
    proposals = proposals.map((proposal) => {
      if (proposal?.key !== proposalKey) return proposal
      updated = updater(proposal)
      return updated || proposal
    })
    if (updated?.axis?.id) {
      emitAxis(updated.axis)
      const axisSlice = sliceForAxis(updated.axis.id)
      if (axisSlice && axisSlice.proposalKey === updated.key) {
        emitSlice(updated, axisSlice.startBin, axisSlice.endBin)
      }
    }
  }

  function clearCalibrationDrag() {
    dragCalibration = null
  }

  function onCalibrationDragStart(e, proposal, bucketIdx, imageId) {
    if (!proposal?.key || !imageId) return
    try { e.stopPropagation() } catch (_) {}
    dragCalibration = {
      proposalKey: proposal.key,
      bucketIdx: Number(bucketIdx),
      imageId: String(imageId),
    }
    try {
      e.dataTransfer.effectAllowed = 'move'
      e.dataTransfer.setData('application/x-axis-calibration', JSON.stringify(dragCalibration))
      e.dataTransfer.setData('text/plain', String(imageId))
    } catch (_) {}
  }

  function onCalibrationDragOver(e) {
    try {
      e.preventDefault()
      e.dataTransfer.dropEffect = 'move'
    } catch (_) {}
  }

  function parseCalibrationDragPayload(e) {
    if (dragCalibration) return dragCalibration
    try {
      const raw = e?.dataTransfer?.getData?.('application/x-axis-calibration')
      if (!raw) return null
      const parsed = JSON.parse(raw)
      if (!parsed || typeof parsed !== 'object') return null
      return {
        proposalKey: String(parsed.proposalKey || ''),
        bucketIdx: Number(parsed.bucketIdx || 0),
        imageId: String(parsed.imageId || ''),
      }
    } catch (_) {
      return null
    }
  }

  function onCalibrationDrop(e, proposal, targetBucketIdx) {
    const payload = parseCalibrationDragPayload(e)
    clearCalibrationDrag()
    if (!payload || !proposal?.key) return
    if (payload.proposalKey !== proposal.key) return
    const fromIdx = Number(payload.bucketIdx)
    const toIdx = Number(targetBucketIdx)
    const imageId = String(payload.imageId || '')
    if (!Number.isFinite(fromIdx) || !Number.isFinite(toIdx) || !imageId) return

    updateProposal(proposal.key, (current) => {
      const calib = current?.calibration
      const currentBuckets = Array.isArray(calib?.buckets) ? calib.buckets : []
      if (currentBuckets.length === 0) return current
      if (toIdx < 0 || toIdx >= currentBuckets.length) return current

      const buckets = currentBuckets.map((b) => ({ ...b, ids: Array.isArray(b?.ids) ? [...b.ids] : [] }))
      for (const b of buckets) b.ids = b.ids.filter((id) => String(id) !== imageId)
      buckets[toIdx].ids.push(imageId)

      const target = clamp01(Number(buckets[toIdx]?.target || 0))
      const axis = {
        ...(current.axis || {}),
        coords: {
          ...((current.axis && current.axis.coords) || {}),
          [imageId]: target,
        },
      }
      return {
        ...current,
        axis,
        calibration: { ...(calib || {}), buckets },
      }
    })
  }

  function apiUrl(path) {
    const base = String(apiBase || '').trim().replace(/\/+$/, '')
    return `${base}${path}`
  }

  function toggleAttributeChip(chipId) {
    attributeChips = (attributeChips || []).map((chip) => {
      if (chip?.id !== chipId) return chip
      const nextSelected = !Boolean(chip.selected)
      return {
        ...chip,
        selected: nextSelected,
        customName: String(chip.customName || chip.name || '').trim(),
      }
    })
  }

  function renameAttributeChip(chipId, value) {
    const nextValue = String(value || '')
    attributeChips = (attributeChips || []).map((chip) => {
      if (chip?.id !== chipId) return chip
      return { ...chip, customName: nextValue }
    })
  }

  function startAttributeChipEdit(chipId) {
    attributeChips = (attributeChips || []).map((chip) => {
      if (chip?.id !== chipId) return { ...chip, editing: false }
      return {
        ...chip,
        editing: true,
        customName: String(chip.customName || chip.name || '').trim(),
        editBackup: String(chip.customName || chip.name || '').trim(),
      }
    })
  }

  function finishAttributeChipEdit(chipId) {
    attributeChips = (attributeChips || []).map((chip) => {
      if (chip?.id !== chipId) return chip
      const nextName = String(chip.customName || chip.name || '').trim() || String(chip.name || '').trim()
      return {
        ...chip,
        editing: false,
        customName: nextName,
        editBackup: '',
      }
    })
  }

  function cancelAttributeChipEdit(chipId) {
    attributeChips = (attributeChips || []).map((chip) => {
      if (chip?.id !== chipId) return chip
      return {
        ...chip,
        editing: false,
        customName: String(chip.editBackup || chip.customName || chip.name || '').trim() || String(chip.name || '').trim(),
        editBackup: '',
      }
    })
  }

  function onAttributeChipKeydown(chipId, e) {
    if (e.key === 'Enter') {
      try { e.preventDefault() } catch (_) {}
      finishAttributeChipEdit(chipId)
    } else if (e.key === 'Escape') {
      try { e.preventDefault() } catch (_) {}
      cancelAttributeChipEdit(chipId)
    }
  }

  function clearPromptWorkflow() {
    promptText = ''
    extractedPromptText = ''
    attributeChips = []
    proposals = []
    recommendations = []
    axisBuildersStore.reset()
    extractionInfo = ''
    recommendationInfo = ''
    errorMsg = ''
    clearSlice()
  }

  function normalizeAttributeType(v) {
    const s = String(v || '').trim().toLowerCase()
    if (s === 'categorical' || s === 'ordinal' || s === 'continuous') return s
    if (s.includes('categor')) return 'categorical'
    if (s.includes('ordin') || s.includes('rank')) return 'ordinal'
    if (s.includes('contin') || s.includes('numeric') || s.includes('scalar')) return 'continuous'
    return ''
  }

  function clamp01(v) {
    const n = Number(v)
    if (!Number.isFinite(n)) return 0
    if (n < 0) return 0
    if (n > 1) return 1
    return n
  }

  function toBinIndexByClientX(clientX, rect, binCount) {
    if (!rect || !Number.isFinite(rect.width) || rect.width <= 0 || binCount <= 0) return 0
    const ratio = (clientX - rect.left) / rect.width
    const clamped = Math.max(0, Math.min(0.999999, ratio))
    return Math.max(0, Math.min(binCount - 1, Math.floor(clamped * binCount)))
  }

  function formatTypeLabel(v) {
    const type = normalizeAttributeType(v)
    if (type === 'categorical') return 'Categorical'
    if (type === 'ordinal') return 'Ordinal'
    if (type === 'continuous') return 'Continuous'
    return 'Attribute'
  }

  function attributeTypeIconClass(v) {
    const type = normalizeAttributeType(v)
    if (type === 'categorical') return 'i-heroicons-squares-2x2'
    if (type === 'ordinal') return 'i-heroicons-bars-3-bottom-left'
    if (type === 'continuous') return 'i-heroicons-arrows-right-left'
    return 'i-heroicons-shape-line'
  }

  function normalizeSupportSpans(spans) {
    if (!Array.isArray(spans)) return []
    const out = []
    for (const raw of spans) {
      if (!raw || typeof raw !== 'object') continue
      const start = Number(raw.start)
      const end = Number(raw.end)
      const text = String(raw.text || '').trim()
      const score = clamp01(Number(raw.score))
      if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start || !text) continue
      out.push({
        start: Math.max(0, Math.floor(start)),
        end: Math.max(0, Math.floor(end)),
        text,
        score,
      })
    }
    return out.sort((a, b) => (a.start - b.start) || (a.end - b.end))
  }

  function buildPromptHighlightSegments(prompt, spans) {
    const text = String(prompt || '')
    if (!text) return []
    const validSpans = normalizeSupportSpans(spans)
      .filter((span) => span.start < text.length && span.end <= text.length)
    if (validSpans.length === 0) {
      return [{ text, active: false, score: 0, start: 0, end: text.length }]
    }

    const boundaries = new Set([0, text.length])
    for (const span of validSpans) {
      boundaries.add(span.start)
      boundaries.add(span.end)
    }
    const points = Array.from(boundaries).sort((a, b) => a - b)
    const segments = []
    for (let i = 0; i < points.length - 1; i += 1) {
      const start = points[i]
      const end = points[i + 1]
      if (end <= start) continue
      const piece = text.slice(start, end)
      const covering = validSpans.filter((span) => span.start <= start && span.end >= end)
      const score = covering.reduce((acc, span) => Math.max(acc, clamp01(span.score)), 0)
      segments.push({
        text: piece,
        active: covering.length > 0,
        score,
        start,
        end,
      })
    }
    return segments
  }

  function compressHistogram(countsRaw, labelsRaw = [], maxBins = HISTOGRAM_MAX_BINS) {
    const counts = Array.isArray(countsRaw) ? countsRaw.map((v) => Number(v) || 0) : []
    const labels = Array.isArray(labelsRaw) ? labelsRaw.map((v) => String(v || '').trim()) : []
    const n = counts.length
    if (n === 0) return { counts: [], maxCount: 1, total: 0, ranges: [], valueLabels: [] }
    const target = Math.max(1, Math.min(Number(maxBins) || 1, n))
    if (target >= n) {
      const ranges = counts.map((_, i) => ({ start: i / n, end: (i + 1) / n }))
      const valueLabels = counts.map((_, i) => labels[i] || '')
      const maxCount = Math.max(1, ...counts)
      const total = counts.reduce((acc, v) => acc + v, 0)
      return { counts, maxCount, total, ranges, valueLabels }
    }
    const out = []
    const ranges = []
    const valueLabels = []
    for (let i = 0; i < target; i += 1) {
      const start = Math.floor((i * n) / target)
      const endExclusive = Math.floor(((i + 1) * n) / target)
      const end = Math.max(start + 1, endExclusive)
      let sum = 0
      for (let j = start; j < end && j < n; j += 1) sum += counts[j]
      out.push(sum)
      ranges.push({ start: start / n, end: Math.min(1, end / n) })
      const bucket = []
      for (let j = start; j < end && j < n; j += 1) {
        const lbl = String(labels[j] || '').trim()
        if (lbl) bucket.push(lbl)
      }
      if (bucket.length === 0) valueLabels.push('')
      else if (bucket.length === 1) valueLabels.push(bucket[0])
      else if (bucket.length <= 3) valueLabels.push(bucket.join(', '))
      else valueLabels.push(`${bucket[0]} ... ${bucket[bucket.length - 1]}`)
    }
    const maxCount = Math.max(1, ...out)
    const total = out.reduce((acc, v) => acc + v, 0)
    return { counts: out, maxCount, total, ranges, valueLabels }
  }

  function histogramBarTooltip(proposal, idx, count) {
    const valueLabels = Array.isArray(proposal?.hist?.valueLabels) ? proposal.hist.valueLabels : []
    const value = String(valueLabels[idx] || '').trim()
    if (value) return `${value}: ${count} images`
    const ranges = Array.isArray(proposal?.hist?.ranges) ? proposal.hist.ranges : []
    const r = ranges[idx]
    if (!r) return `${count} images`
    const lo = `${Math.round(clamp01(r.start) * 100)}%`
    const hi = `${Math.round(clamp01(r.end) * 100)}%`
    return `${lo} to ${hi}: ${count} images`
  }

  function idsForSlice(proposal, startBin, endBin) {
    const axis = proposal?.axis
    const coords = axis?.coords || {}
    const binCount = Math.max(1, Number(proposal?.hist?.counts?.length) || 1)
    const lo = Math.min(startBin, endBin)
    const hi = Math.max(startBin, endBin)
    const ids = []
    for (const [id, raw] of Object.entries(coords)) {
      const v = clamp01(raw)
      const idx = Math.max(0, Math.min(binCount - 1, Math.floor(v * binCount)))
      if (idx >= lo && idx <= hi) ids.push(id)
    }
    return ids
  }

  function emitSlice(proposal, startBin, endBin) {
    const binCount = Math.max(1, Number(proposal?.hist?.counts?.length) || 1)
    const lo = Math.min(startBin, endBin)
    const hi = Math.max(startBin, endBin)
    const ids = idsForSlice(proposal, lo, hi)
    updateActiveSlice({
      proposalKey: proposal.key,
      axisId: proposal.axis.id,
      axisName: proposal.axis.name || proposal.axis.id,
      startBin: lo,
      endBin: hi,
      binCount,
      ids,
    })
  }

  function isSliceBarActive(proposal, idx) {
    const axisId = String(proposal?.axis?.id || '').trim()
    const axisSlice = sliceForAxis(axisId)
    if (!axisSlice || axisSlice.proposalKey !== proposal?.key) return false
    return idx >= axisSlice.startBin && idx <= axisSlice.endBin
  }

  function onHistogramPointerDown(e, proposal) {
    const binCount = Math.max(1, Number(proposal?.hist?.counts?.length) || 1)
    const rect = e.currentTarget?.getBoundingClientRect?.()
    if (!rect || binCount <= 0) return
    const idx = toBinIndexByClientX(e.clientX, rect, binCount)
    dragSlice = { proposalKey: proposal.key, rect, binCount, startBin: idx }
    emitSlice(proposal, idx, idx)
    window.addEventListener('pointermove', onWindowPointerMove)
    window.addEventListener('pointerup', onWindowPointerUp, { once: true })
    try { e.preventDefault() } catch (_) {}
  }

  function onWindowPointerMove(e) {
    if (!dragSlice || !Array.isArray(proposals)) return
    const proposal = proposals.find((p) => p.key === dragSlice.proposalKey)
    if (!proposal) return
    const idx = toBinIndexByClientX(e.clientX, dragSlice.rect, dragSlice.binCount)
    emitSlice(proposal, dragSlice.startBin, idx)
  }

  function onWindowPointerUp() {
    dragSlice = null
    window.removeEventListener('pointermove', onWindowPointerMove)
  }

  onDestroy(() => {
    window.removeEventListener('pointermove', onWindowPointerMove)
  })

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

  function sessionFromAxisResponse(chip, data) {
    const axis = data?.axis
    if (!axis?.id || !axis?.coords) return null
    return {
      axisId: axis.id,
      q: String(chip?.customName || chip?.name || data?.q || axis?.name || '').trim(),
      axis: {
        ...axis,
        name: String(chip?.customName || chip?.name || axis?.name || '').trim() || axis.name,
      },
      ids: Array.isArray(data?.ids) ? data.ids.map((v) => String(v || '').trim()) : [],
      projectionValues: Array.isArray(data?.projection_values) ? data.projection_values.map((v) => Number(v) || 0) : [],
      projectionMin: Number(data?.projection_min || 0),
      projectionMax: Number(data?.projection_max || 0),
      scores: Array.isArray(data?.scores) ? data.scores.map((v) => Number(v) || 0) : [],
      std: Array.isArray(data?.std) ? data.std.map((v) => Number(v) || 0) : [],
      decileExemplars: Array.isArray(data?.decile_exemplars) ? data.decile_exemplars : [],
      hotspots: Array.isArray(data?.hotspots) ? data.hotspots : [],
      moveCount: Number(data?.move_count || 0),
      maxMoves: Number(data?.max_moves || 0),
      moves: Array.isArray(data?.moves) ? data.moves : [],
      undefinedIds: Array.isArray(data?.undefined_ids) ? data.undefined_ids.map((v) => String(v || '').trim()).filter(Boolean) : [],
      w0Summary: data?.w0_summary || {},
      moveHistory: Array.isArray(chip?.moveHistory) ? chip.moveHistory : [],
    }
  }

  function currentSubsetIds() {
    const normalized = normalizedSliceArray(activeSlices)
      .map((slice) => ({
        ...slice,
        ids: Array.isArray(slice.ids) ? slice.ids.map((id) => String(id || '').trim()).filter(Boolean) : [],
      }))
      .filter((slice) => slice.ids.length > 0)
    if (normalized.length > 0) {
      let intersection = new Set(normalized[0].ids)
      for (let i = 1; i < normalized.length; i += 1) {
        const set = new Set(normalized[i].ids)
        intersection = new Set(Array.from(intersection).filter((id) => set.has(id)))
      }
      return Array.from(intersection)
    }
    return (items || []).map((item) => String(item?.id || '').trim()).filter(Boolean)
  }

  function formatPct(v) {
    const n = Number(v)
    if (!Number.isFinite(n)) return '0.0'
    return (n * 100).toFixed(1)
  }

  async function recommendScatterplots() {
    errorMsg = ''
    recommendationInfo = ''
    recommendations = []
    if (!Array.isArray(candidateAxes) || candidateAxes.length < 2) {
      errorMsg = 'Need at least two axes before requesting recommendations.'
      return
    }
    recommendingScatterplots = true
    try {
      const data = await postJson('/analysis/recommend_scatterplots', {
        dataset: dataset || undefined,
        axes: candidateAxes,
        subset_ids: currentSubsetIds(),
        top_k: 10,
        min_overlap: 24,
        k_neighbors: 12,
      })
      const recs = Array.isArray(data?.recommendations) ? data.recommendations : []
      recommendations = recs
      if (recs.length === 0) recommendationInfo = 'No recommended axis pair was found with enough overlap.'
      else recommendationInfo = `Recommended ${recs.length} axis pairs using correlation and metadata separability.`
    } catch (e2) {
      recommendations = []
      recommendationInfo = ''
      errorMsg = `Recommendation failed: ${String(e2)}`
    } finally {
      recommendingScatterplots = false
    }
  }

  function applyRecommendationPair(rec) {
    const xId = String(rec?.x_axis_id || '').trim()
    const yId = String(rec?.y_axis_id || '').trim()
    if (!xId || !yId) return
    dispatch('setX', { id: xId })
    dispatch('setY', { id: yId })
  }

  function normalizeExtractedAttributes(extraction) {
    const attrs = Array.isArray(extraction?.attributes) ? extraction.attributes : []
    const out = []
    for (let i = 0; i < attrs.length; i += 1) {
      const entry = attrs[i]
      let name = ''
      let type = ''
      if (entry && typeof entry === 'object') {
        name = String(entry?.name || entry?.attribute || entry?.label || '').trim()
        type = normalizeAttributeType(entry?.type || entry?.attribute_type || entry?.kind)
      } else {
        name = String(entry || '').trim()
      }
      if (!name) continue
      out.push({
        id: `chip:${i}:${name.toLowerCase().replace(/\s+/g, '-')}`,
        name,
        type,
        selected: false,
        customName: name,
        editing: false,
        editBackup: '',
        supportPhrases: Array.isArray(entry?.support_phrases) ? entry.support_phrases.map((v) => String(v || '').trim()).filter(Boolean) : [],
        supportSpans: normalizeSupportSpans(entry?.support_spans),
      })
    }
    return out
  }

  function proposalFromDistribution(chip, data, idx) {
    const variableType = normalizeAttributeType(data?.attribute_type || chip?.type || '')
    const axisRaw = data?.axis
    if (!axisRaw?.id || !axisRaw?.coords) return null
    const renamed = String(chip?.customName || '').trim()
    const fallbackName = String(chip?.name || axisRaw?.name || '').trim()
    const targetName = renamed || fallbackName
    const ids = Array.isArray(data?.ids) ? data.ids.map((v) => String(v || '').trim()) : []
    const predictedValues = Array.isArray(data?.predicted_values) ? data.predicted_values : []
    const predictedById = {}
    for (let i = 0; i < Math.min(ids.length, predictedValues.length); i += 1) {
      const id = String(ids[i] || '').trim()
      const label = String(predictedValues[i] || '').trim()
      if (!id || !label) continue
      predictedById[id] = label
    }
    const values = Array.isArray(data?.values) ? data.values.map((v) => String(v || '').trim()).filter(Boolean) : []
    const axis = {
      ...axisRaw,
      name: targetName || axisRaw.name,
      attribute_type: variableType || axisRaw?.attribute_type || '',
      scoring_method: data?.scoring_method || axisRaw?.scoring_method || '',
    }
    let histCounts = Array.isArray(data?.score_histogram?.counts)
      ? data.score_histogram.counts.map((v) => Number(v) || 0)
      : []
    let histLabels = []
    if (variableType === 'categorical' || variableType === 'ordinal') {
      const valueHist = Array.isArray(data?.value_histogram) ? data.value_histogram : []
      const counts = valueHist.map((row) => Number(row?.count) || 0)
      const labels = valueHist.map((row) => String(row?.value || '').trim())
      if (counts.length > 0) histCounts = counts
      if (labels.some((s) => Boolean(s))) histLabels = labels
    }
    if (histLabels.length === 0 && Array.isArray(axis?.labels)) {
      histLabels = axis.labels.map((v) => String(v || '').trim())
    }
    const hist = compressHistogram(histCounts, histLabels, HISTOGRAM_MAX_BINS)
    const baseProposal = {
      key: `${axis.id}:${idx}`,
      variableType,
      variableTypeLabel: formatTypeLabel(variableType),
      axis,
      hist,
      values,
      predictedById,
    }
    return {
      ...baseProposal,
      calibration: buildCalibration(baseProposal),
    }
  }

  async function extractAttributes() {
    const prompt = String(promptText || '').trim()
    errorMsg = ''
    extractionInfo = ''
    clearSlice()
    if (!prompt) {
      errorMsg = 'Enter a visualization prompt first.'
      return
    }
    if (!Array.isArray(items) || items.length === 0) {
      errorMsg = 'Load images before extracting attributes.'
      return
    }

    extracting = true
    try {
      const extraction = await postJson('/llm/extract_attributes', {
        prompt,
        dataset: dataset || undefined,
        max_attributes: 14,
      })
      const chips = normalizeExtractedAttributes(extraction)
      if (chips.length === 0) {
        throw new Error('LLM returned no dimensions.')
      }
      attributeChips = chips
      extractedPromptText = prompt
      proposals = []
      extractionInfo = `LLM proposed ${chips.length} dimensions. Select chips, rename if needed, then create axes.`
    } catch (e) {
      extractedPromptText = ''
      attributeChips = []
      proposals = []
      clearSlice()
      errorMsg = `LLM extraction failed: ${String(e)}`
    } finally {
      extracting = false
    }
  }

  async function createAxesFromSelectedChips() {
    const prompt = String(promptText || '').trim()
    errorMsg = ''
    clearSlice()
    if (!prompt) {
      errorMsg = 'Enter a visualization prompt first.'
      return
    }
    if (!Array.isArray(selectedChips) || selectedChips.length === 0) {
      errorMsg = 'Select at least one dimension chip.'
      return
    }
    if (!Array.isArray(items) || items.length === 0) {
      errorMsg = 'Load images before creating axes.'
      return
    }

    creatingAxes = true
    try {
      const results = await Promise.all(
        selectedChips.map(async (chip) => {
          const renamed = String(chip?.customName || '').trim()
          const query = renamed || String(chip?.name || '').trim()
          const data = await postJson('/axis/create', {
            collection_id: dataset || undefined,
            dataset: dataset || undefined,
            q: query,
          })
          return { chip: { ...chip, customName: query }, data }
        })
      )

      const nextBuilders = results
        .map(({ chip, data }) => sessionFromAxisResponse(chip, data))
        .filter(Boolean)

      if (nextBuilders.length === 0) {
        throw new Error('No valid axis builders were produced for selected chips.')
      }
      axisBuildersStore.replace(nextBuilders)
      proposals = []
      for (const builder of nextBuilders) {
        if (builder?.axis?.id) emitAxis(builder.axis)
      }
      extractionInfo = ''
    } catch (e) {
      axisBuildersStore.reset()
      proposals = []
      clearSlice()
      errorMsg = `Axis creation failed: ${String(e)}`
    } finally {
      creatingAxes = false
    }
  }

  async function addManualAxis() {
    const query = String(manualAxisText || '').trim()
    errorMsg = ''
    clearSlice()
    if (!query) {
      errorMsg = 'Enter an axis name first.'
      return
    }
    if (!Array.isArray(items) || items.length === 0) {
      errorMsg = 'Load images before creating axes.'
      return
    }

    creatingAxes = true
    try {
      const data = await postJson('/axis/create', {
        collection_id: dataset || undefined,
        dataset: dataset || undefined,
        q: query,
      })
      const builder = sessionFromAxisResponse({ name: query, customName: query }, data)
      if (!builder) {
        throw new Error('No valid axis builder was produced.')
      }
      axisBuildersStore.upsert(builder)
      if (builder?.axis?.id) emitAxis(builder.axis)
      manualAxisText = ''
    } catch (e) {
      errorMsg = `Axis creation failed: ${String(e)}`
    } finally {
      creatingAxes = false
    }
  }

  async function onAxisBuilderMove(e) {
    const axisId = String(e?.detail?.axisId || '').trim()
    const imageId = String(e?.detail?.imageId || '').trim()
    const newScore0To100 = Number(e?.detail?.newScore0To100 || 0)
    const fromScore0To100 = Number(e?.detail?.fromScore0To100 || 0)
    const moveType = String(e?.detail?.moveType || 'score').trim().toLowerCase() || 'score'
    if (!axisId || !imageId) return

    errorMsg = ''
    clearSlice()

    const current = (axisBuilders || []).find((entry) => entry?.axisId === axisId)
    try {
      const data = await postJson('/axis/move', {
        axis_id: axisId,
        image_id: imageId,
        new_score_0_100: newScore0To100,
        move_type: moveType,
      })
      const updated = sessionFromAxisResponse(current || {}, data)
      if (!updated) throw new Error('Invalid axis move response')
      const nextHistory = Array.isArray(current?.moveHistory) ? [...current.moveHistory] : []
      nextHistory.unshift({
        imageId,
        fromScore0To100,
        toScore0To100: newScore0To100,
      })
      updated.moveHistory = nextHistory.slice(0, 6)
      axisBuildersStore.upsert(updated)
      if (updated?.axis?.id) emitAxis(updated.axis)
    } catch (err) {
      errorMsg = `Axis move failed: ${String(err)}`
    }
  }

  function onAxisBuilderRemove(e) {
    const axisId = String(e?.detail?.axisId || '').trim()
    if (!axisId) return
    axisBuildersStore.remove(axisId)
    removeSliceForAxis(axisId)
    emitAxisRemoval(axisId)
  }

  function onAxisBuilderSave(e) {
    const axisId = String(e?.detail?.axisId || '').trim()
    const axis = e?.detail?.axis
    const q = String(e?.detail?.q || axis?.name || '').trim()
    if (!axisId || !axis) return
    dispatch('saveAxis', {
      axisId,
      axisName: String(axis?.name || q || axisId).trim(),
      q,
      dataset: dataset || '',
      modelType: String(axis?.model_type || '').trim(),
    })
  }
</script>

<div class="prompt-sidebar">
  <div class="subtile">
    <textarea
      id="prompt-input"
      class="prompt-input"
      bind:value={promptText}
      placeholder="What do you want to visualize?"
    />
    <div class="mt-2 flex items-center gap-2">
      <button class="btn btn-primary btn-xs" disabled={extracting || creatingAxes} on:click={extractAttributes}>
        {extracting ? 'Analyzing...' : 'Analyze'}
      </button>
      <button class="btn btn-ui-secondary btn-xs" disabled={extracting || creatingAxes} on:click={clearPromptWorkflow}>
        Clear
      </button>
      {#if activeSlices.length > 0}
        <button class="btn btn-ui-secondary btn-xs" on:click={clearSlice}>
          Unslice
        </button>
      {/if}
    </div>
    {#if errorMsg}
      <div class="mt-2 text-sm text-red-600">{errorMsg}</div>
    {/if}

    {#if extractedPromptText}
      <div class="mt-2 prompt-support-card">
        <div class="prompt-support-label">
          {#if hoveredExplainChip}
            {String(hoveredExplainChip.customName || hoveredExplainChip.name || '').trim()}
          {:else}
            Prompt support
          {/if}
        </div>
        <div class="prompt-support-text">
          {#each promptSupportSegments as segment, idx (`segment:${idx}:${segment.start}`)}
            <span
              class={`prompt-support-segment ${segment.active ? 'active' : ''}`}
              style={segment.active ? `--support-alpha:${(0.12 + (segment.score * 0.28)).toFixed(3)};` : ''}
            >{segment.text}</span>
          {/each}
        </div>
      </div>
    {/if}

    {#if attributeChips.length > 0}
      <div class="mt-2 chip-grid">
        {#each attributeChips as chip (chip.id)}
          <div
            role="group"
            class={`dim-chip ${chip.selected ? 'selected' : ''} ${hoveredExplainChipId === chip.id ? 'explaining' : ''}`}
            on:mouseenter={() => { hoveredExplainChipId = chip.id }}
            on:mouseleave={() => { if (hoveredExplainChipId === chip.id) hoveredExplainChipId = '' }}
            on:focusin={() => { hoveredExplainChipId = chip.id }}
            on:focusout={() => { if (hoveredExplainChipId === chip.id) hoveredExplainChipId = '' }}
          >
            <div class="dim-chip-row">
              <button
                type="button"
                class="dim-chip-toggle"
                on:click={() => toggleAttributeChip(chip.id)}
                title={chip.selected ? 'Deselect dimension' : 'Select dimension'}
              >
                <span class="truncate">{String(chip.customName || chip.name || '').trim() || chip.name}</span>
              </button>
              <div class="dim-chip-tools">
                <button
                  type="button"
                class="dim-chip-edit"
                aria-label="Rename suggested attribute"
                  title={chip.supportPhrases?.length ? `Rename suggested attribute. Support: ${chip.supportPhrases.join(' | ')}` : 'Rename suggested attribute'}
                  on:click|stopPropagation={() => startAttributeChipEdit(chip.id)}
                >
                  <span class="i-heroicons-pencil-square" />
                </button>
                <span
                  class={`axis-type-icon ${attributeTypeIconClass(chip.type)}`}
                  role="img"
                  aria-label={formatTypeLabel(chip.type)}
                  title={formatTypeLabel(chip.type)}
                />
              </div>
            </div>
            {#if chip.editing}
              <input
                class="dim-chip-rename"
                type="text"
                value={chip.customName}
                on:input={(e) => renameAttributeChip(chip.id, e.currentTarget.value)}
                on:blur={() => finishAttributeChipEdit(chip.id)}
                on:keydown={(e) => onAttributeChipKeydown(chip.id, e)}
                placeholder="Rename axis"
              />
            {/if}
          </div>
        {/each}
      </div>
      <div class="mt-2 flex items-center justify-between gap-3">
        <button
          class="btn btn-primary btn-xs"
          disabled={creatingAxes || extracting || selectedChipCount === 0}
          on:click={createAxesFromSelectedChips}
        >
          {creatingAxes ? 'Creating...' : 'Create axes'}
        </button>
        <div class="text-xs text-slate-500">{selectedChipCount}/{attributeChips.length}</div>
      </div>
    {/if}
  </div>

  <div class="subtile">
    <div class="flex items-center justify-between gap-2">
      <div class="sidebar-section-title">Axes</div>
      <div class="text-xs text-slate-500">{axisBuilders.length}</div>
    </div>
    <div class="axis-entry-row mt-2">
      <input
        class="manual-axis-input"
        type="text"
        bind:value={manualAxisText}
        placeholder="Add an axis directly"
        on:keydown={(e) => {
          if (e.key === 'Enter') {
            try { e.preventDefault() } catch (_) {}
            addManualAxis()
          }
        }}
      />
      <button
        class="btn btn-primary btn-xs"
        disabled={creatingAxes || extracting || !String(manualAxisText || '').trim()}
        on:click={addManualAxis}
      >
        Add
      </button>
    </div>

    {#if axisBuilders.length === 0}
      <div class="mt-1" />
    {:else}
      <div class="mt-2 grid gap-2">
        {#each axisBuilders as builder (builder.axisId)}
          <AxisBuilder
            session={builder}
            itemsById={itemById}
            {selectedX}
            {selectedY}
            activeSlice={sliceForAxis(builder.axisId)}
            busy={false}
            on:move={onAxisBuilderMove}
            on:sliceChange={(e) => {
              const nextSlice = e.detail?.slice || null
              if (nextSlice) updateActiveSlice(nextSlice)
              else removeSliceForAxis(builder.axisId)
            }}
            on:useX={(e) => {
              if (e.detail?.axis?.id) emitAxis(e.detail.axis)
              if (e.detail?.axis?.id) dispatch('setX', { id: e.detail.axis.id })
            }}
            on:useY={(e) => {
              if (e.detail?.axis?.id) emitAxis(e.detail.axis)
              if (e.detail?.axis?.id) dispatch('setY', { id: e.detail.axis.id })
            }}
            on:remove={onAxisBuilderRemove}
            on:save={onAxisBuilderSave}
            on:openImage={(e) => {
              const imageId = String(e.detail?.imageId || '').trim()
              if (!imageId) return
              dispatch('openImage', { imageId })
            }}
          />
        {/each}
      </div>
    {/if}
  </div>

  <div class="sidebar-bottom">
    <div class="subtile">
      <div class="mt-2 flex items-center gap-2">
        <button
          class="btn btn-primary btn-xs"
          disabled={recommendingScatterplots || candidateAxes.length < 2}
          on:click={recommendScatterplots}
        >
          {recommendingScatterplots ? 'Ranking...' : 'Rank pairs'}
        </button>
        <div class="text-xs text-slate-500">{currentSubsetIds().length} imgs</div>
      </div>

      {#if recommendationInfo}
        <div class="mt-2 text-xs text-slate-600">{recommendationInfo}</div>
      {/if}
      {#if recommendations.length > 0}
        <div class="mt-2 recommendation-list">
          {#each recommendations as rec, idx}
            <div class="recommendation-item">
              <div class="recommendation-title">{idx + 1}. {rec.x_axis_name} × {rec.y_axis_name}</div>
              <div class="recommendation-metrics">
                {formatPct(rec.score)} score · {formatPct(rec.separability_score)} sep · {formatPct(rec.correlation_score)} corr
              </div>
              {#if rec.best_metadata_field}
                <div class="recommendation-meta">
                  meta: {rec.best_metadata_field}
                </div>
              {/if}
              <button class="btn btn-xs btn-ui-secondary mt-1" on:click={() => applyRecommendationPair(rec)}>
                Use
              </button>
            </div>
          {/each}
        </div>
      {/if}
    </div>
  </div>

  {#if showLoadingBackdrop}
    <div class="sidebar-loading-backdrop" role="status" aria-live="polite" aria-label={loadingBackdropText}>
      <div class="sidebar-loading-card">
        <span class="sidebar-loading-spinner" aria-hidden="true" />
        <span>{loadingBackdropText}</span>
      </div>
    </div>
  {/if}
</div>

<style>
  .prompt-sidebar {
    position: relative;
    height: 100%;
    overflow: auto;
    padding-right: 4px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .sidebar-bottom {
    margin-top: auto;
  }

  .prompt-input {
    width: 100%;
    margin: 6px 0 0;
    min-height: 88px;
    resize: vertical;
    padding: 10px 12px;
    font-size: var(--font-size-body);
    line-height: 1.4;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    background: #ffffff;
    color: #0f172a;
  }

  .prompt-input::placeholder {
    color: #94a3b8;
  }

  .chip-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 6px;
  }

  .prompt-support-card {
    border: 1px solid #dbe2ec;
    border-radius: 8px;
    background: #ffffff;
    padding: 8px 10px;
    display: grid;
    gap: 6px;
  }

  .prompt-support-label {
    font-size: var(--font-size-small);
    color: #64748b;
  }

  .prompt-support-text {
    font-size: var(--font-size-body);
    line-height: 1.5;
    color: #0f172a;
    white-space: pre-wrap;
  }

  .prompt-support-segment {
    border-radius: 4px;
    transition: background-color 120ms ease, box-shadow 120ms ease;
  }

  .prompt-support-segment.active {
    background: rgba(96, 165, 250, var(--support-alpha, 0.18));
    box-shadow: inset 0 -1px 0 rgba(59, 130, 246, 0.18);
  }

  .dim-chip {
    border: 1px solid #dbe2ec;
    border-radius: 8px;
    background: #fff;
    padding: 6px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .dim-chip.selected {
    border-color: #93c5fd;
    box-shadow: inset 0 0 0 1px #bfdbfe;
    background: #f8fbff;
  }

  .dim-chip.explaining {
    border-color: #60a5fa;
    box-shadow: inset 0 0 0 1px rgba(96, 165, 250, 0.28);
  }

  .dim-chip-row {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .dim-chip-toggle {
    display: flex;
    align-items: center;
    min-width: 0;
    flex: 1;
    margin: 0;
    padding: 0;
    border: 0;
    background: transparent;
    text-align: left;
    color: #0f172a;
  }

  .dim-chip-tools {
    display: flex;
    align-items: center;
    gap: 6px;
    flex: none;
  }

  .dim-chip-edit {
    display: grid;
    place-items: center;
    width: 24px;
    height: 24px;
    border: 1px solid #dbe2ec;
    border-radius: 6px;
    background: #ffffff;
    color: #64748b;
  }

  .axis-type-icon {
    flex: none;
    font-size: 0.92rem;
    color: #64748b;
  }

  .dim-chip-rename {
    margin: 0;
    width: 100%;
    padding: 4px 6px;
    font-size: var(--font-size-body);
    border-radius: 6px;
    border: 1px solid #cbd5e1;
    color: #0f172a;
    background: #ffffff;
  }

  .axis-entry-row {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .manual-axis-input {
    flex: 1;
    min-width: 0;
    border: 1px solid #d5dde8;
    border-radius: 6px;
    padding: 6px 8px;
    background: #ffffff;
    color: #0f172a;
    font-size: var(--font-size-body);
  }

  .axis-proposal {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 8px;
    background: #fff;
    cursor: grab;
  }

  .axis-proposal:active {
    cursor: grabbing;
  }

  .histogram {
    height: 70px;
    border: 1px solid #dbeafe;
    background: transparent;
    display: grid;
    align-items: end;
    gap: 1px;
    padding: 8px 6px 6px;
    cursor: col-resize;
    user-select: none;
  }

  .histogram-bar {
    display: block;
    border-radius: 0;
    background: #cbd5e1;
    min-height: 2px;
    opacity: 0.9;
    transition: opacity 120ms ease, background-color 120ms ease;
  }

  .histogram-bar:hover {
    opacity: 1;
    background: #94a3b8;
  }

  .histogram-bar.active {
    opacity: 1;
    background: #38bdf8;
  }

  .axis-calibration {
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 6px;
    background: #ffffff;
  }

  .axis-calibration-title {
    font-size: var(--font-size-small);
    color: #475569;
    margin-bottom: 4px;
  }

  .calibration-grid {
    display: grid;
    gap: 4px;
    overflow-x: auto;
    padding-bottom: 2px;
  }

  .calibration-bucket {
    border: 1px solid #dbe2ec;
    border-radius: 4px;
    background: #f8fafc;
    min-height: 74px;
    padding: 3px;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .calibration-label {
    font-size: var(--font-size-small);
    color: #334155;
    line-height: 1.1;
    text-align: center;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .calibration-items {
    flex: 1;
    display: grid;
    place-items: center;
    min-height: 54px;
  }

  .calibration-thumb {
    width: 40px;
    height: 40px;
    object-fit: cover;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    cursor: grab;
    background: #ffffff;
  }

  .calibration-thumb:active {
    cursor: grabbing;
  }

  .calibration-empty {
    font-size: var(--font-size-small);
    color: #94a3b8;
  }

  .calibration-fallback {
    font-size: var(--font-size-small);
    color: #475569;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 4px;
    background: #fff;
    cursor: grab;
    max-width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .recommendation-list {
    display: grid;
    gap: 6px;
  }

  .recommendation-item {
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 6px;
    background: #ffffff;
  }

  .recommendation-title {
    font-size: var(--font-size-body);
    font-weight: 600;
    color: #0f172a;
  }

  .recommendation-metrics {
    margin-top: 2px;
    font-size: var(--font-size-small);
    color: #475569;
  }

  .recommendation-meta {
    margin-top: 2px;
    font-size: var(--font-size-small);
    color: #64748b;
  }

  .sidebar-section-title {
    font-size: var(--font-size-title);
    font-weight: 700;
    color: #334155;
    line-height: 1.1;
  }

  .sidebar-loading-backdrop {
    position: absolute;
    inset: 0;
    z-index: 120;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255, 255, 255, 0.78);
    backdrop-filter: blur(1px);
  }

  .sidebar-loading-card {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 9px 12px;
    border: 1px solid #dbe2ec;
    border-radius: 10px;
    background: #ffffff;
    color: #334155;
    font-size: var(--font-size-body);
    font-weight: 600;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.12);
  }

  .sidebar-loading-spinner {
    width: 14px;
    height: 14px;
    border-radius: 999px;
    border: 2px solid #cbd5e1;
    border-top-color: #334155;
    animation: sidebar-spin 0.8s linear infinite;
  }

  @keyframes sidebar-spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>
