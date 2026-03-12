<script>
  import { createEventDispatcher, onDestroy } from 'svelte'

  export let session = null
  export let itemsById = new Map()
  export let selectedX = null
  export let selectedY = null
  export let busy = false
  export let activeSlice = null

  const dispatch = createEventDispatcher()
  const SLICE_SEGMENTS = 24
  const MIN_SLICE_WIDTH_PCT = 10
  const KDE_WIDTH = 500
  const KDE_HEIGHT = 120
  let dragSlice = null
  let localActiveSlice = null
  let hoveredDecileIndex = null
  let previewDecileIndex = null
  let dropDecileIndex = null
  let dropUndefinedActive = false
  let openPriorPromptSide = null // 'neg' | 'pos' | null

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

  function formatRawValue(v) {
    const n = Number(v)
    if (!Number.isFinite(n)) return '0'
    const abs = Math.abs(n)
    if (abs >= 100) return n.toFixed(0)
    if (abs >= 10) return n.toFixed(1)
    if (abs >= 1) return n.toFixed(2)
    return n.toFixed(3).replace(/\.?0+$/, '')
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
      }
    } catch (_) {
      return null
    }
  }

  function allowDrop(e) {
    try {
      e.preventDefault()
      e.dataTransfer.dropEffect = 'move'
    } catch (_) {}
  }

  function onDragStart(e, item) {
    if (!item?.id || busy) return
    try {
      e.dataTransfer.effectAllowed = 'move'
      e.dataTransfer.setData('application/x-axis-builder-item', JSON.stringify({
        imageId: item.id,
        score: Number(item.score_0_100 || 0),
      }))
      e.dataTransfer.setData('text/plain', String(item.id))
    } catch (_) {}
  }

  function decileTargetPct(index) {
    const idx = Math.max(0, Math.min(9, Number(index) || 0))
    return (idx * 10) + 5
  }

  function dispatchMove(imageId, fromScore0To100, targetPct, moveType = 'score') {
    dispatch('move', {
      axisId: session?.axis?.id || session?.axisId,
      imageId,
      newScore0To100: clamp100(targetPct),
      fromScore0To100: clamp100(fromScore0To100),
      moveType,
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
    dispatchMove(payload.imageId, payload.score, decileTargetPct(idx), 'score')
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
    dispatchMove(payload.imageId, payload.score, payload.score, 'undefined')
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

  function clearSlice() {
    localActiveSlice = null
    dragSlice = null
    window.removeEventListener('pointermove', onWindowPointerMove)
    dispatch('sliceChange', { slice: null })
  }

  function clientPct(e, rect) {
    if (!rect || !Number.isFinite(rect.width) || rect.width <= 0) return 0
    return clamp01((Number(e.clientX || 0) - rect.left) / rect.width) * 100
  }

  function onPlotPointerDown(e) {
    const rect = e.currentTarget?.getBoundingClientRect?.()
    if (!rect) return
    const startDisplayPct = clientPct(e, rect)
    dragSlice = { rect, startDisplayPct, lastDisplayPct: startDisplayPct }
    window.addEventListener('pointermove', onWindowPointerMove)
    window.addEventListener('pointerup', onWindowPointerUp, { once: true })
    try { e.preventDefault() } catch (_) {}
  }

  function onWindowPointerMove(e) {
    if (!dragSlice) return
    dragSlice.lastDisplayPct = clientPct(e, dragSlice.rect)
    const nextSlice = buildSlice(dragSlice.startDisplayPct, dragSlice.lastDisplayPct)
    if (nextSlice) commitSlice(nextSlice)
  }

  function onWindowPointerUp() {
    if (dragSlice) {
      const nextSlice = buildSlice(dragSlice.startDisplayPct, dragSlice.lastDisplayPct)
      if (nextSlice) commitSlice(nextSlice)
    }
    dragSlice = null
    window.removeEventListener('pointermove', onWindowPointerMove)
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

  function decileColumns(deciles) {
    const rows = Array.isArray(deciles) ? deciles : []
    return Array.from({ length: 10 }, (_, idx) => {
      const row = rows.find((entry) => Number(entry?.bin_index || 0) === idx) || {}
      const rawExemplars = Array.isArray(row?.exemplars) ? row.exemplars : []
      const sortedExemplars = [...rawExemplars].sort((a, b) => (Number(a?.std) || 0) - (Number(b?.std) || 0))
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
  }

  function onMarkerLeave(index) {
    const idx = Math.max(0, Math.min(9, Number(index) || 0))
    if (previewDecileIndex === idx) previewDecileIndex = null
    onDecileLeave(idx)
  }

  function priorPromptList(side) {
    const summary = session?.w0Summary || {}
    const raw = side === 'neg' ? summary?.neg_prompts : summary?.pos_prompts
    if (!Array.isArray(raw)) return []
    return raw.map((v) => String(v || '').trim()).filter((v) => v.length > 0)
  }

  function togglePriorPromptMenu(side) {
    openPriorPromptSide = (openPriorPromptSide === side) ? null : side
  }

  function closePriorPromptMenu() {
    openPriorPromptSide = null
  }

  function openImage(imageId) {
    const id = String(imageId || '').trim()
    if (!id) return
    dispatch('openImage', { imageId: id })
  }

  onDestroy(() => {
    window.removeEventListener('pointermove', onWindowPointerMove)
  })

  $: scoreEntries = (() => {
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

  $: undefinedIds = new Set(Array.isArray(session?.undefinedIds) ? session.undefinedIds.map((v) => String(v || '').trim()).filter(Boolean) : [])
  $: rawEntries = scoreEntries.filter((entry) => !undefinedIds.has(entry.id))
  $: rawValuesSorted = rawEntries.map((entry) => Number(entry.rawValue || 0)).filter((value) => Number.isFinite(value)).sort((a, b) => a - b)
  $: rawMin = rawValuesSorted.length > 0 ? rawValuesSorted[0] : 0
  $: rawMax = rawValuesSorted.length > 0 ? rawValuesSorted[rawValuesSorted.length - 1] : 1
  $: rawMid = rawMin + ((rawMax - rawMin) * 0.5)
  $: definedStdValues = rawEntries.map((entry) => Number(entry.std || 0)).filter((value) => Number.isFinite(value))
  $: densityPlot = buildDensityPlot(rawEntries.map((entry) => Number(entry.rawValue || 0)))
  $: uncertaintyTrend = buildUncertaintyTrend(
    rawEntries.map((entry) => Number(entry.rawValue || 0)),
    definedStdValues,
    densityPlot.width,
    densityPlot.height,
  )
  $: exampleDeciles = decileColumns(session?.decileExemplars)
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
  $: movedIds = new Set(Array.isArray(session?.moves) ? session.moves.map((move) => String(move?.image_id || '')) : [])
  $: undefinedEntries = Array.from(undefinedIds)
    .map((id) => {
      const idx = scoreEntries.findIndex((entry) => entry.id === id)
      if (idx < 0) return null
      return {
        id,
        score_0_100: Number(scoreEntries[idx]?.score_0_100 || 0),
        rawValue: Number(scoreEntries[idx]?.rawValue || 0),
        std: Number(scoreEntries[idx]?.std || 0),
      }
    })
    .filter(Boolean)
  $: previewDecile = Number.isInteger(previewDecileIndex) ? (exampleDeciles[previewDecileIndex] || null) : null
  $: previewMarker = Number.isInteger(previewDecileIndex) ? (decileMarkers.find((marker) => marker.index === previewDecileIndex) || null) : null
  $: negPriorPrompts = priorPromptList('neg')
  $: posPriorPrompts = priorPromptList('pos')
  $: activePriorPrompts = openPriorPromptSide === 'neg' ? negPriorPrompts : (openPriorPromptSide === 'pos' ? posPriorPrompts : [])
  $: {
    const axisId = session?.axis?.id || session?.axisId
    if (activeSlice && activeSlice?.axisId === axisId) {
      localActiveSlice = activeSlice
    } else if (!dragSlice) {
      localActiveSlice = null
    }
  }
</script>

<article class="axis-builder-card">
  <div class="axis-builder-header">
    <div class="axis-builder-title">{session?.axis?.name || session?.q || 'Axis'}</div>
    <div class="axis-builder-actions">
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
      >S</button>
      <button
        type="button"
        class={`axis-action ${selectedX === session?.axis?.id ? 'active x' : ''}`}
        on:click={() => dispatch('useX', { axis: session?.axis })}
        aria-label="Use axis as X"
      >X</button>
      <button
        type="button"
        class={`axis-action ${selectedY === session?.axis?.id ? 'active y' : ''}`}
        on:click={() => dispatch('useY', { axis: session?.axis })}
        aria-label="Use axis as Y"
      >Y</button>
      <button
        type="button"
        class="axis-action axis-remove"
        on:click={() => dispatch('remove', { axisId: session?.axis?.id || session?.axisId })}
        aria-label="Remove axis"
        title="Remove axis"
      >×</button>
    </div>
  </div>

  <div class="density-shell">
    {#if localActiveSlice}
      <div
        class="density-slice-overlay"
        style={`left:${clamp100(localActiveSlice.rangeStartPct || 0)}%;width:${Math.max(0.8, clamp100(localActiveSlice.rangeEndPct || 0) - clamp100(localActiveSlice.rangeStartPct || 0))}%;`}
      />
    {/if}
    <svg
      class="density-plot"
      viewBox={`0 0 ${densityPlot.width} ${densityPlot.height}`}
      preserveAspectRatio="none"
      role="presentation"
      on:pointerdown|stopPropagation={onPlotPointerDown}
      on:dblclick|stopPropagation={clearSlice}
    >
      <path class="density-line" d={densityPlot.linePath} />
      {#if uncertaintyTrend.bandPath}
        <path class="uncert-band" d={uncertaintyTrend.bandPath} />
      {/if}
      {#if uncertaintyTrend.meanPath}
        <path class="uncert-line" d={uncertaintyTrend.meanPath} />
      {/if}
    </svg>
    <div class="density-legend" aria-hidden="true">
      <span class="density-legend-item">
        <span class="density-legend-line density-legend-line-dist" />
        <span>distribution</span>
      </span>
      <span class="density-legend-item">
        <span class="density-legend-line density-legend-line-uncert" />
        <span>uncertainty</span>
      </span>
    </div>
    <div class="density-dot-layer">
      {#each decileMarkers as marker (marker.index)}
        <button
          type="button"
          class={`density-dot ${hoveredDecileIndex === marker.index || dropDecileIndex === marker.index ? 'active' : ''}`}
          style={`left:${marker.xPct}%;top:${(marker.y / Math.max(1, densityPlot.height)) * 100}%;`}
          aria-label={`${exampleDeciles[marker.index]?.label || ''} decile`}
          on:pointerdown|stopPropagation={() => {}}
          on:mouseenter={() => onMarkerEnter(marker.index)}
          on:mouseleave={() => onMarkerLeave(marker.index)}
          on:focus={() => onMarkerEnter(marker.index)}
          on:blur={() => onMarkerLeave(marker.index)}
        />
      {/each}
    </div>
    {#if previewDecile && previewMarker && Array.isArray(previewDecile.samples) && previewDecile.samples.length > 0}
      <div
        class="decile-preview-pop"
        aria-hidden="true"
        style={`left:${previewMarker.xPct}%;top:calc(${(previewMarker.y / Math.max(1, densityPlot.height)) * 100}% + 10px);`}
      >
        {#each previewDecile.samples as sample (sample.id)}
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
    <button
      type="button"
      class={`prior-end-btn prior-end-btn-neg ${openPriorPromptSide === 'neg' ? 'active' : ''}`}
      aria-label="Inspect negative prior prompts"
      title="Inspect negative prior prompts"
      disabled={negPriorPrompts.length === 0}
      on:pointerdown|stopPropagation={() => {}}
      on:click|stopPropagation={() => togglePriorPromptMenu('neg')}
    >−</button>
    <button
      type="button"
      class={`prior-end-btn prior-end-btn-pos ${openPriorPromptSide === 'pos' ? 'active' : ''}`}
      aria-label="Inspect positive prior prompts"
      title="Inspect positive prior prompts"
      disabled={posPriorPrompts.length === 0}
      on:pointerdown|stopPropagation={() => {}}
      on:click|stopPropagation={() => togglePriorPromptMenu('pos')}
    >+</button>
    {#if openPriorPromptSide}
      <div
        class={`prior-popup ${openPriorPromptSide === 'neg' ? 'left' : 'right'}`}
        role="dialog"
        aria-label={openPriorPromptSide === 'neg' ? 'Negative prior prompts' : 'Positive prior prompts'}
        on:pointerdown|stopPropagation
      >
        <div class="prior-popup-head">
          <span>{openPriorPromptSide === 'neg' ? 'Negative prior prompts' : 'Positive prior prompts'}</span>
          <button
            type="button"
            class="prior-popup-close"
            aria-label="Close prior prompt list"
            on:click|stopPropagation={closePriorPromptMenu}
          >×</button>
        </div>
        {#if activePriorPrompts.length > 0}
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

  <div class="builder-density-axis" aria-hidden="true">
    <span>{formatRawValue(rawMin)}</span>
    <span>{formatRawValue(rawMid)}</span>
    <span>{formatRawValue(rawMax)}</span>
  </div>

  <div class="decile-ribbons">
    <div class="decile-label-row" aria-hidden="true">
      <span class="decile-label-spacer" />
      {#each exampleDeciles as decile (decile.index)}
        <span class="decile-label">{decile.index === 9 ? '100' : String(decile.index * 10)}</span>
      {/each}
    </div>

    <div class="decile-row-wrap">
      <span class="decile-row-label">most sure</span>
      <div class="decile-row" role="list" aria-label="Most certain examples by decile">
        {#each exampleDeciles as decile (decile.index)}
          {@const exemplar = decile.mostCertain}
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
              on:dragstart={(e) => { if (exemplar) onDragStart(e, exemplar) }}
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
    </div>

    <div class="decile-row-wrap">
      <span class="decile-row-label">least sure</span>
      <div class="decile-row" role="list" aria-label="Least certain examples by decile">
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
              on:dragstart={(e) => { if (exemplar) onDragStart(e, exemplar) }}
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
    </div>

    <div class="decile-row-wrap undefined-row-wrap">
      <span class="decile-row-label">undefined</span>
      <div
        class={`undefined-row ${dropUndefinedActive ? 'drop-active' : ''}`}
        role="list"
        aria-label="Undefined examples for this axis"
        on:dragover={allowDrop}
        on:dragenter={onUndefinedDropEnter}
        on:dragleave={onUndefinedDropLeave}
        on:drop={onUndefinedDrop}
      >
        {#each undefinedEntries as exemplar (exemplar.id)}
          {@const item = resolveItem(exemplar.id)}
          <div role="listitem" class="undefined-thumb-wrap">
            <button
              type="button"
              class="decile-thumb-btn undefined-thumb-btn"
              draggable={!busy}
              on:dragstart={(e) => onDragStart(e, exemplar)}
              on:click|stopPropagation={() => { if (exemplar?.id) openImage(exemplar.id) }}
              title={exemplar?.id || 'Undefined'}
            >
              {#if item?.thumbUrl || item?.url}
                <img src={item.thumbUrl || item.url} alt={exemplar?.id || 'Undefined'} class="decile-thumb" />
              {:else}
                <div class="decile-thumb decile-thumb-empty" aria-hidden="true" />
              {/if}
            </button>
          </div>
        {/each}
      </div>
    </div>
  </div>
</article>

<style>
  .axis-builder-card {
    border: 1px solid #d9d9dd;
    border-radius: 14px;
    background: #ffffff;
    padding: 12px 12px 10px;
    display: grid;
    gap: 8px;
  }

  .axis-builder-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
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

  .axis-builder-actions {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    flex: none;
  }

  .axis-action {
    width: 28px;
    height: 28px;
    border: 1px solid #d5d8e7;
    border-radius: 9px;
    background: #fbfbfd;
    color: #71758b;
    font-size: var(--font-size-body);
    line-height: 1;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    transition: border-color 120ms ease, color 120ms ease, background-color 120ms ease;
  }

  .axis-action.active {
    border-color: #b6b8ff;
    background: #f6f6ff;
    color: #6267db;
  }

  .axis-action.axis-remove {
    color: #d46d7f;
    border-color: #efc8d0;
    background: #fff7f8;
  }

  .axis-action.axis-save {
    color: #475569;
    border-color: #d7dde8;
    background: #ffffff;
  }

  .density-shell {
    position: relative;
    border-top: 1px solid #ececf2;
    padding-top: 6px;
    overflow: visible;
  }

  .density-slice-overlay {
    position: absolute;
    top: 6px;
    bottom: 0;
    background: rgba(96, 165, 250, 0.16);
    border-left: 1px solid rgba(96, 165, 250, 0.38);
    border-right: 1px solid rgba(96, 165, 250, 0.38);
    pointer-events: none;
    z-index: 1;
  }

  .density-plot {
    position: relative;
    z-index: 2;
    display: block;
    width: 100%;
    height: 126px;
    user-select: none;
    cursor: col-resize;
  }

  .density-line {
    fill: none;
    stroke: #6a67db;
    stroke-width: 2.3;
    stroke-linecap: round;
    stroke-linejoin: round;
  }

  .uncert-band {
    fill: rgba(16, 185, 129, 0.16);
    stroke: none;
  }

  .uncert-line {
    fill: none;
    stroke: #0f9b75;
    stroke-width: 1.8;
    stroke-linecap: round;
    stroke-linejoin: round;
    stroke-dasharray: 4 3;
  }

  .density-legend {
    position: absolute;
    bottom: 8px;
    left: 10px;
    z-index: 4;
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 4px 6px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.9);
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
    color: #636a7f;
    font-size: 10px;
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
    color: #6a67db;
  }

  .density-legend-line-uncert {
    color: #0f9b75;
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
    width: 7px;
    height: 7px;
    border: 0;
    border-radius: 999px;
    background: #6a67db;
    cursor: pointer;
    pointer-events: auto;
    transition: transform 120ms ease, background-color 120ms ease, box-shadow 120ms ease;
    transform: translate(-50%, -50%);
    box-shadow: 0 0 0 1px #6a67db;
  }

  .density-dot.active {
    background: #5e5aca;
    box-shadow: 0 0 0 1px #5e5aca;
  }

  .decile-preview-pop {
    position: absolute;
    z-index: 5;
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

  .decile-preview-thumb {
    width: 48px;
    height: 48px;
    border-radius: 8px;
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
    position: absolute;
    bottom: 8px;
    width: 20px;
    height: 20px;
    border: 1px solid #cfd2df;
    border-radius: 999px;
    background: #ffffff;
    color: #6a67db;
    font-size: 12px;
    font-weight: 700;
    line-height: 1;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    z-index: 4;
  }

  .prior-end-btn:disabled {
    opacity: 0.45;
    color: #8b8fa8;
  }

  .prior-end-btn.active {
    border-color: #7d7adf;
    background: #f7f7ff;
  }

  .prior-end-btn-neg {
    left: 2px;
  }

  .prior-end-btn-pos {
    right: 2px;
  }

  .prior-popup {
    position: absolute;
    top: 10px;
    width: min(320px, calc(100% - 8px));
    max-height: 130px;
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

  .prior-popup-close {
    border: 0;
    background: transparent;
    color: #7e8397;
    font-size: 14px;
    line-height: 1;
    padding: 0;
    width: 16px;
    height: 16px;
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

  .builder-density-axis {
    display: flex;
    justify-content: space-between;
    margin-top: -1px;
    color: #8a8da2;
    font-size: var(--font-size-small);
    line-height: 1;
  }

  .decile-ribbons {
    display: grid;
    gap: 4px;
    overflow-x: auto;
    padding-bottom: 4px;
    scrollbar-width: thin;
  }

  .decile-label-row {
    display: flex;
    gap: 4px;
    min-width: max-content;
    margin-bottom: 1px;
    align-items: center;
  }

  .decile-label-spacer {
    width: 58px;
    flex: none;
  }

  .decile-label {
    width: 40px;
    flex: none;
    text-align: center;
    color: #8a8da2;
    font-size: var(--font-size-small);
    line-height: 1;
  }

  .decile-row {
    display: flex;
    gap: 4px;
    min-width: max-content;
  }

  .undefined-row {
    min-height: 40px;
    flex: 1 1 auto;
    display: flex;
    gap: 4px;
    align-items: center;
    flex-wrap: wrap;
    padding: 2px 0;
    border-radius: 8px;
    min-width: 0;
  }

  .decile-row-wrap {
    display: flex;
    gap: 4px;
    align-items: center;
    min-width: max-content;
  }

  .decile-row-label {
    width: 58px;
    flex: none;
    display: inline-flex;
    align-items: center;
    justify-content: flex-end;
    height: 40px;
    padding-right: 2px;
    color: #7b8197;
    font-size: var(--font-size-small);
    line-height: 1;
  }

  .decile-slot {
    width: 40px;
    height: 40px;
    flex: none;
    border-radius: 7px;
    position: relative;
  }

  .decile-slot.drop-active {
    outline: 2px solid rgba(106, 103, 219, 0.35);
    outline-offset: 1px;
  }

  .undefined-row.drop-active {
    background: rgba(106, 103, 219, 0.08);
    box-shadow: inset 0 0 0 1px rgba(106, 103, 219, 0.24);
  }

  .undefined-thumb-wrap {
    width: 40px;
    height: 40px;
    flex: none;
  }

  .decile-thumb-btn {
    width: 100%;
    height: 100%;
    border: 0;
    border-radius: 7px;
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

  .undefined-thumb-btn {
    width: 40px;
    height: 40px;
    flex: none;
  }

  .decile-thumb-btn.moved .decile-thumb {
    box-shadow: 0 0 0 2px rgba(106, 103, 219, 0.42);
  }

  .decile-thumb {
    width: 100%;
    height: 100%;
    display: block;
    object-fit: cover;
    user-select: none;
    -webkit-user-drag: none;
    border-radius: 7px;
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
    .decile-label-row,
    .decile-row,
    .decile-row-wrap {
      min-width: max-content;
    }

    .decile-label-spacer,
    .decile-row-label {
      width: 52px;
    }

    .decile-row-label {
      height: 34px;
    }

    .decile-label,
    .decile-slot {
      width: 34px;
      height: 34px;
    }
  }
</style>
