<script>
  // import ImageGrid from './components/ImageGrid.svelte'
  import { generateDemoImages, sortBySimilarity, clusterKMeans } from './lib/data'
  import { onMount, tick } from 'svelte'
  import AxesMinimap from './components/AxesMinimap.svelte'
  import AxesPanel from './components/AxesPanel.svelte'
  import ConceptsPanel from './components/ConceptsPanel.svelte'
  
  // import ConceptComposer from './components/ConceptComposer.svelte'
  // import CombinedConceptItem from './components/CombinedConceptItem.svelte'
  import Callout from './components/Callout.svelte'
  import CombinedSelection from './components/CombinedSelection.svelte'
  import SelectionComposer from './components/SelectionComposer.svelte'
  
  let allImages = []
  let prevImages = []
  // Cache of gallery responses by key (dataset|embed|method)
  const galleryCache = new Map()
  let selected = new Set()
  
  // Backend wiring
  // const API_BASE = (import.meta.env && import.meta.env.VITE_API_BASE) ? import.meta.env.VITE_API_BASE : 'http://127.0.0.1:5001'
  const API_BASE = (import.meta.env && import.meta.env.VITE_API_BASE) ? import.meta.env.VITE_API_BASE : 'http://localhost:5002'
  console.log('[frontend] API_BASE', API_BASE)
  let datasetPath = '' // leave blank to let server pick sample
  let datasets = []

  // Radio selection and effective embed method
  let embedSelection = 'color_rgb' // 'color' | 'clip' | 'shape' | 'meta' | custom
  let embedMethod = 'color_rgb' // backend method: 'color_rgb' | 'clip' | 'dino' | 'dift_sd_partXY'
  let warningMsg = ''
  // External label storage (object, separate from images)
  let labelDB = {}
  $: posCount = Object.values(labelDB).filter(v => v === 'pos').length
  $: negCount = Object.values(labelDB).filter(v => v === 'neg').length

  // Scribble controls
  let scribbleLabel = 'pos' // 'pos' | 'neg'
  let minimapContainerRef
  
  // Dynamic minimap size based on available space
  $: minimapW = (() => {
    void leftWidth; void windowWidth; void leftTileH;
    const w = minimapContainerRef ? Math.floor(minimapContainerRef.clientWidth -470) : 700
    return Math.max(300, Math.min(2000, w))
  })()

  $: minimapH = (() => {
    void leftTileH;
    const h = minimapContainerRef ? Math.floor(minimapContainerRef.clientHeight - 50) : 700
    return Math.max(300, Math.min(2000, h))
  })()
  
  // Axes state
  let axes = [] // [{ id, name, coords }]
  let selectedAxisX = null
  let selectedAxisY = null
  // Custom projections created by user
  // Shape: { id: 'custom:<ts>', name, xAxisId, yAxisId }
  let customProjections = []

  // Debugging: inspect an axis and report how many image ids it covers and sample values
  function logAxisDebug(axisId, which = '') {
    try {
      const ax = (axes || []).find(a => a.id === axisId)
      if (!ax) { console.warn('[frontend] axis not found', which, axisId); return }
      const ids = (allImages || []).map(i => i.id)
      const coordKeys = Object.keys(ax.coords || {})
      console.log('[frontend] axis keys', which, { id: ax.id, keysCount: coordKeys.length, keysSample: coordKeys.slice(0, 5) })
      let matched = 0
      const samples = []
      for (let i = 0; i < Math.min(ids.length, 50); i++) {
        const id = ids[i]
        const v = ax.coords ? ax.coords[id] : undefined
        if (typeof v === 'number' && isFinite(v)) {
          matched++
          if (samples.length < 5) samples.push({ id, v })
        }
      }
      // Count total matches across all ids
      let totalMatched = 0
      for (const id of ids) {
        const v = ax.coords ? ax.coords[id] : undefined
        if (typeof v === 'number' && isFinite(v)) totalMatched++
      }
      console.log('[frontend] axis debug', which, { id: ax.id, name: ax.name, group: ax.group, totalMatched, totalImages: ids.length, samples })
    } catch (err) {
      console.warn('[frontend] axis debug error', which, axisId, err)
    }
  }
  
  function resetAllStateForDatasetChange() {
    // Clear in-memory state
    allImages = []
    prevImages = []
    selected = new Set()
    axes = []
    selectedAxisX = null
    selectedAxisY = null
    customProjections = []
    selections = []
    combinedSelections = []
    labelDB = {}
    warningMsg = ''
    // Clear caches
    galleryCache.clear()
    // Clear persisted keys
    try {
      localStorage.removeItem('promptherder.concepts')
      localStorage.removeItem('promptherder.combined')
      
      localStorage.removeItem('promptherder.axes')
      localStorage.removeItem('promptherder.customProjections')
      localStorage.removeItem('promptherder.selections')
      localStorage.removeItem('promptherder.combinedSelections')
    } catch (_) {}
  }

  async function onDatasetSelect(val) {
    if (val === undefined) return
    // If the same value, ignore
    if ((datasetPath || '') === (val || '')) return
    datasetPath = val
    resetAllStateForDatasetChange()
    await loadGallery()
  }

  // Split pane state
  let leftWidth = 1200
  let leftTileH = 920
  let leftCollapsed = false

  let middleWidth = 700
  let middleTileH = 600
  let middleCollapsed = true

  let rightWidth = 320 // default ~w-80
  let rightTileH = 600
  let rightCollapsed = true

  let minLeft = 500 // tune: minimum visible width for the left pane
  let maxLeft = 2000 // tune: maximum visible width for the left pane
  let minRight = 180 // tune: minimum right panel width

  // Corner tile drag resize state
  let tileDrag = null // { which: 'left'|'middle'|'right', startX, startY, startW, startH }
  function startLeftTileResize(e) {
    tileDrag = { which: 'left', startX: e.clientX, startY: e.clientY, startW: leftWidth, startH: leftTileH }
  }
  function startMiddleTileResize(e) {
    tileDrag = { which: 'middle', startX: e.clientX, startY: e.clientY, startW: middleWidth, startH: middleTileH }
  }
  function startRightTileResize(e) {
    tileDrag = { which: 'right', startX: e.clientX, startY: e.clientY, startW: rightWidth, startH: rightTileH }
  }
  
  let windowWidth = 0
  const projectionOptions = [
    { value: 'color_rgb', label: 'Color' },
    { value: 'clip', label: 'Semantic' },
    { value: 'shape', label: 'Shape' },
    { value: 'meta', label: 'Metadata' },
  ]
  function onDrag(e) {
    if (tileDrag) {
      const dx = e.clientX - tileDrag.startX
      const dy = e.clientY - tileDrag.startY
      if (tileDrag.which === 'left') {
        leftWidth = Math.max(minLeft, Math.min(maxLeft, tileDrag.startW + dx))
        leftTileH = Math.max(300, tileDrag.startH + dy)
      } else if (tileDrag.which === 'middle') {
        middleWidth = Math.max(400, tileDrag.startW + dx)
        middleTileH = Math.max(300, tileDrag.startH + dy)
      } else if (tileDrag.which === 'right') {
        rightWidth = Math.max(minRight, tileDrag.startW + dx)
        rightTileH = Math.max(300, tileDrag.startH + dy)
      }
    }
  }
  function endDrag() {
    tileDrag = null
  }

  // Create or update the two default axes for the current projection (x,y)
  function axisPrefixForMethod(method) {
    if (!method) return 'Axis'
    if (method === 'color_rgb') return 'RGB'
    if (method === 'color_hsv') return 'HSV'
    if (method === 'color_lch') return 'LCh'
    if (method === 'clip') return 'CLIP'
    if (method === 'dino') return 'DINO'
    if (method.startsWith('dift_sd_part')) return `DIFT ${method.replace('dift_sd_part','')}`
    if (method === 'dift_sd') return 'DIFT'
    return method
  }

  function ensureDefaultAxesForCurrentProjection() {
    const methodName = embedMethod
    if (!methodName) return
    const prefix = axisPrefixForMethod(methodName)
    const idX = `axis:${methodName}:x`
    const idY = `axis:${methodName}:y`
    const coordsX = {}
    const coordsY = {}
    for (const it of allImages) {
      coordsX[it.id] = Number(it.x ?? 0)
      coordsY[it.id] = Number(it.y ?? 0)
    }
    // Group by projection-level key (color_rgb, shape, clip, or custom id if selected)
    const projGroup = (embedSelection && String(embedSelection).startsWith('custom:')) ? embedSelection : (embedSelection || 'color_rgb')
    let next = axes
    const axX = { id: idX, name: `${prefix} X`, coords: coordsX, group: projGroup }
    const axY = { id: idY, name: `${prefix} Y`, coords: coordsY, group: projGroup }
    const hasX = next.some(a => a.id === idX)
    const hasY = next.some(a => a.id === idY)
    if (hasX) next = next.map(a => a.id === idX ? axX : a); else next = [...next, axX]
    if (hasY) next = next.map(a => a.id === idY ? axY : a); else next = [...next, axY]
    axes = next
    // Default selection: take first two axes under this projection group
    const underGroup = axes.filter(a => (a?.group === projGroup) || idMatchesProjection(projGroup, a?.id || ''))
    selectedAxisX = underGroup[0]?.id || idX
    selectedAxisY = underGroup[1]?.id || idY
  }

  function idMatchesProjection(groupKey, axisId) {
    if (!axisId || !groupKey) return false
    if (groupKey === 'color_rgb') return axisId.startsWith('axis:color_lch:') || axisId.startsWith('axis:color_hsv:') || axisId.startsWith('axis:color_rgb:')
    if (groupKey === 'shape') return axisId.startsWith('axis:dino:') || axisId.startsWith('axis:dift_sd_part') || axisId.startsWith('axis:dift_sd:')
    if (groupKey === 'clip') return axisId.startsWith('axis:clip:')
    if (groupKey === 'meta') return axisId.startsWith('axis:meta:')
    return false
  }

  // Selections state
  let selections = []
  let combinedSelections = []

  const THUMB_SIZE = 200
  function toThumbUrl(u) {
    if (!u) return ''
    // Map '/images/rel' -> `/thumb/<THUMB_SIZE>/rel`
    if (u.startsWith('/images/')) {
      return `/thumb/${THUMB_SIZE}${u.substring('/images'.length)}`
    }
    return u
  }
  function prefixUrl(u) {
    if (!u) return ''
    if (u.startsWith('http://') || u.startsWith('https://')) return u
    // Prefer thumbnails for local image paths
    const maybeThumb = toThumbUrl(u)
    if (maybeThumb.startsWith('/')) return API_BASE + maybeThumb
    return u
  }

  // Lightweight prefetch of thumbnails to keep them hot in memory
  const _prefetched = new Set()
  function schedulePrefetch(urls) {
    if (!Array.isArray(urls) || urls.length === 0) return
    // Stagger prefetch using idle time to avoid blocking UI
    const run = () => {
      let count = 0
      for (const u of urls) {
        if (!u || _prefetched.has(u)) continue
        try {
          const img = new Image()
          img.decoding = 'async'
          img.loading = 'eager'
          img.referrerPolicy = 'no-referrer'
          img.src = u
          _prefetched.add(u)
          count++
          if (count >= 12) break // limit per tick
        } catch (_) {}
      }
      // If there are more to prefetch, schedule another tick
      const remaining = urls.filter(u => u && !_prefetched.has(u))
      if (remaining.length > 0) setTimeout(run, 80)
    }
    if ('requestIdleCallback' in window) {
      try { window.requestIdleCallback(run, { timeout: 300 }) } catch (_) { setTimeout(run, 50) }
    } else {
      setTimeout(run, 50)
    }
  }

  async function loadGallery() {
    const qs = new URLSearchParams()
    if (datasetPath && datasetPath.trim()) qs.set('dataset', datasetPath.trim())
    if (embedMethod) qs.set('embed', embedMethod)
    // Use PCA for stable, deterministic 2D coordinates
    qs.set('method', 'pca')

    const cacheKey = `${datasetPath || ''}|${embedMethod}|pca`
    if (galleryCache.has(cacheKey)) {
      console.log('[frontend] cache hit', cacheKey)
      const cached = galleryCache.get(cacheKey)
      warningMsg = ''
      prevImages = allImages
      allImages = cached.items
      try { schedulePrefetch(allImages.map(it => it.url)) } catch (_) {}
      // Populate axes from current projection (x,y)
      ensureDefaultAxesForCurrentProjection()
      // Merge cached metadata axes if any
      try {
        const metaAxes = Array.isArray(cached.metaAxes) ? cached.metaAxes : []
        console.log('[frontend] cache metaAxes count', metaAxes.length)
        if (metaAxes.length > 0) {
          const incoming = metaAxes.map(a => ({ id: a.id, name: a.name || a.id, coords: a.coords || {}, labels: a.labels || [], label_positions: a.label_positions || [], group: 'meta' }))
          const existing = new Map(axes.map(a => [a.id, a]))
          for (const ax of incoming) { if (!existing.has(ax.id)) existing.set(ax.id, ax) }
          axes = Array.from(existing.values())
        }
      } catch (err) { console.warn('[frontend] failed to merge cached metaAxes', err) }
      return
    }

    try {
      const url = `${API_BASE}/gallery.json?${qs.toString()}`
      console.log('[frontend] fetch', url)
      const res = await fetch(url)
      if (!res.ok) throw new Error('Failed to fetch gallery')
      const data = await res.json()
      console.log('[frontend] response items', data?.items?.length ?? 0, 'n_layer', data?.n_layer)
      if (!data || !Array.isArray(data.items)) throw new Error('Invalid gallery payload')

      warningMsg = data.warning || ''

      // Save previous state for animation before replacing
      prevImages = allImages
      allImages = data.items.map((it) => ({
        id: it.id,
        url: prefixUrl(it.url || it.path),
        className: it.className || 'Unknown',
        label: it.id,
        gx: Number(it.gx ?? it.x ?? 0),
        gy: Number(it.gy ?? it.y ?? 0),
        x: Number(it.x ?? it.gx ?? 0),
        y: Number(it.y ?? it.gy ?? 0),
        embed: [Number(it.gx ?? it.x ?? 0), Number(it.gy ?? it.y ?? 0)],
      }))
      // Cache for quick toggling between embeddings (include metadata axes)
      const metaAxesRaw = Array.isArray(data.metadata_axes) ? data.metadata_axes : []
      console.log('[frontend] fetched metaAxes', metaAxesRaw)
      galleryCache.set(cacheKey, { items: allImages, metaAxes: metaAxesRaw })
      try { schedulePrefetch(allImages.map(it => it.url)) } catch (_) {}
      // Populate axes from current projection (x,y)
      ensureDefaultAxesForCurrentProjection()
      // Ingest metadata axes if present
      try {
        const metaAxes = Array.isArray(data.metadata_axes) ? data.metadata_axes : []
        if (metaAxes.length > 0) {
          const imgIds = (allImages || []).map(i => i.id)
          function urlParts(u) {
            try {
              let path = u || ''
              const idx = path.indexOf('://')
              if (idx > -1) {
                const slash3 = path.indexOf('/', idx + 3)
                path = slash3 > -1 ? path.substring(slash3) : path
              }
              const m = path.match(/\/thumb\/(\d+)\/(.*)$/)
              const rel = m ? m[2] : (path.startsWith('/images/') ? path.substring('/images/'.length) : path)
              const base = rel.split('/').pop() || rel
              const dot = base.lastIndexOf('.')
              const stem = dot>0 ? base.substring(0, dot) : base
              return { rel, base, stem }
            } catch (_) { return { rel: '', base: '', stem: '' } }
          }
          // Build variant -> imageId map for robust matching
          const variantToId = new Map()
          for (const it of (allImages || [])) {
            const id = it.id
            const info = urlParts(it.url || '')
            const variants = new Set([
              String(id), String(id).toLowerCase(),
              info.rel, info.rel.toLowerCase(),
              info.base, info.base.toLowerCase(),
              info.stem, info.stem.toLowerCase(),
            ])
            for (const v of Array.from(variants).filter(Boolean)) {
              if (!variantToId.has(v)) variantToId.set(v, id)
              // also map with backslashes/slashes swapped
              const swap = v.replace(/\\/g,'/').replace(/\//g,'\\')
              if (swap && !variantToId.has(swap)) variantToId.set(swap, id)
            }
          }
          function normalizeMetaAxis(raw) {
            const coords = raw.coords || {}
            const out = {}
            let matched = 0
            const keyList = Object.keys(coords)
            for (const k of keyList) {
              let v = coords[k]
              // Coerce numeric strings to numbers
              if (!(typeof v === 'number')) {
                const num = Number(v)
                if (Number.isFinite(num)) v = num
              }
              if (!(typeof v === 'number' && isFinite(v))) continue
              const cand = [k, k.toLowerCase()]
              // also add stem of k
              const base = k.split('/').pop() || k
              const dot = base.lastIndexOf('.')
              const stem = dot>0 ? base.substring(0, dot) : base
              cand.push(base, base.toLowerCase(), stem, stem.toLowerCase())
              let targetId = null
              for (const c of cand) { if (c && variantToId.has(c)) { targetId = variantToId.get(c); break } }
              if (targetId) { if (out[targetId] === undefined) { out[targetId] = v; matched++ } }
            }
            console.log('[frontend] normalize meta axis', raw.id, raw.name, 'matched', matched, '/', imgIds.length, 'keys', keyList.length)
            if (matched === 0) {
              console.warn('[frontend] meta axis produced zero matches; sample keys', keyList.slice(0,5))
            }
            // Pass through labels info for axis tick rendering
            const labels = Array.isArray(raw.labels) ? raw.labels : []
            const label_positions = Array.isArray(raw.label_positions) ? raw.label_positions : []
            return { id: raw.id, name: raw.name || raw.id, coords: out, labels, label_positions, group: 'meta' }
          }
          const incoming = metaAxes.map(normalizeMetaAxis)
          const existing = new Map(axes.map(a => [a.id, a]))
          for (const ax of incoming) { if (!existing.has(ax.id)) existing.set(ax.id, ax) }
          axes = Array.from(existing.values())
        } else {
          console.log('[frontend] no metadata axes in response')
        }
      } catch (err) { console.warn('[frontend] meta ingest error', err) }
    } catch (e) {
      console.error('[frontend] fetch error', e)
      // Fallback to demo data
      warningMsg = ''
      allImages = generateDemoImages(48)
    }
  }

  onMount(() => {
    // Try to fetch datasets list for the dropdown; fallback to placeholders above
    ;(async () => {
      try {
        const res = await fetch(`${API_BASE}/datasets`)
        if (res.ok) {
          const data = await res.json()
          if (Array.isArray(data)) {
            // Accept an array of strings or objects
            const mapped = data.map((d) => {
              if (typeof d === 'string') return { label: d, value: d }
              if (d && typeof d === 'object') {
                const label = d.label || d.name || d.id || d.path || 'Dataset'
                const value = d.value || d.id || d.path || d.name || label
                return { label, value }
              }
              return null
            }).filter(Boolean)
            if (mapped.length > 0) datasets = mapped
          }
        }
      } catch (_) { /* ignore, keep placeholders */ }
    })()
    try {
      
      const rawAxes = localStorage.getItem('promptherder.axes')
      if (rawAxes) {
        const parsedA = JSON.parse(rawAxes)
        if (Array.isArray(parsedA)) axes = parsedA
      }
      const rawCustom = localStorage.getItem('promptherder.customProjections')
      if (rawCustom) {
        const parsedCP = JSON.parse(rawCustom)
        if (Array.isArray(parsedCP)) customProjections = parsedCP
      }
      const rawSel = localStorage.getItem('promptherder.selections')
      if (rawSel) {
        const parsedS = JSON.parse(rawSel)
        if (Array.isArray(parsedS)) selections = parsedS
      }
      const rawCombSel = localStorage.getItem('promptherder.combinedSelections')
      if (rawCombSel) {
        const parsedCS = JSON.parse(rawCombSel)
        if (Array.isArray(parsedCS)) combinedSelections = parsedCS
      }
    } catch (e) { /* ignore */ }
    loadGallery()
  })
  
  $: (function persistAxes(a) {
    try { localStorage.setItem('promptherder.axes', JSON.stringify(a)) } catch (_) {}
  })(axes)
  $: (function persistCustomProjections(c) {
    try { localStorage.setItem('promptherder.customProjections', JSON.stringify(c)) } catch (_) {}
  })(customProjections)
  $: (function persistSelections(s) {
    try { localStorage.setItem('promptherder.selections', JSON.stringify(s)) } catch (_) {}
  })(selections)
  $: (function persistCombinedSelections(s) {
    try { localStorage.setItem('promptherder.combinedSelections', JSON.stringify(s)) } catch (_) {}
  })(combinedSelections)

  function onEmbedChange() {
    warningMsg = ''
    // Compute effective method for each selection
    if (embedSelection === 'meta') {
      // Do not change embedMethod or reload; just pick first two metadata axes
      const underGroup = axes.filter(a => (a?.group === 'meta') || idMatchesProjection('meta', a?.id || ''))
      console.log('[frontend] selecting meta projection; available meta axes', underGroup.map(a=>a.id))
      if (underGroup.length >= 2) {
        selectedAxisX = underGroup[0].id
        selectedAxisY = underGroup[1].id
        console.log('[frontend] set meta axes X/Y', selectedAxisX, selectedAxisY)
      } else {
        console.warn('[frontend] meta selection but fewer than 2 axes found')
      }
      return
    } else if (embedSelection === 'shape') {
      // default shape method is dino; DIFT selection overrides embedMethod in handler
      embedMethod = 'dino'
    } else if (embedSelection === 'color_rgb') {
      embedMethod = 'color_rgb'
    } else {
      embedMethod = embedSelection
    }
    // Switching to a normal embed; reload gallery
    loadGallery()
  }

  function onScribbleLabel(e) {
    const { ids, label, updates } = e.detail || {}
    if (Array.isArray(updates)) {
      const next = { ...labelDB }
      for (const u of updates) {
        if (!u) continue
        const id = u.id
        const v = u.label
        if (v === null || v === undefined || v === 'none') delete next[id]
        else if (v === 'pos' || v === 'neg') next[id] = v
      }
      labelDB = next
      return
    }
    if (Array.isArray(ids)) {
      const next = { ...labelDB }
      for (const id of ids) {
        if (label === null || label === undefined || label === 'none') delete next[id]
        else next[id] = label
      }
      labelDB = next
    }
  }
  
</script>

<svelte:window bind:innerWidth={windowWidth} on:mousemove={onDrag} on:mouseup={endDrag} />

<div class="app-main text-gray-900">
  <header class="app-header">
    <div class="app-header-inner">
      <div class="i-heroicons-sparkles brand-icon" />
      <div class="brand-name">ReQuest</div>
      <!-- Dataset selector -->
      <div class="ml-4 inline-flex items-center gap-2">
        <label for="dataset-select" class="text-sm text-gray-700">Dataset</label>
        <select id="dataset-select" class="text-sm"
                on:change={(e)=> onDatasetSelect(e.currentTarget.value)}>
          {#each datasets as d}
            <option value={d.value} selected={(datasetPath||'')===(d.value||'')}>{d.label}</option>
          {/each}
        </select>
      </div>
      <div class="flex-1" />
    </div>
  </header>

  {#if warningMsg}
    <div class="bg-yellow-50 border-l-4 border-yellow-400 text-yellow-800 p-3">
      <div class="container mx-auto px-4 text-sm">
        {warningMsg} — run: <code>python backend/precompute_embeddings.py {datasetPath || '[DATASET_PATH]'} --methods color_rgb,clip,dino,dift_sd</code>
      </div>
    </div>
  {/if}

  <main class="app-main w-full py-6">
    <div class="panels-row flex flex-nowrap overflow-x-auto" >
      <!-- Left minimap + list -->
      {#if leftCollapsed}
        <div class="shrink-0" style="width:50px;">
          <button
            class="collapsed-handle rotate tile-header"
            title="Define projection"
            on:click={() => { leftCollapsed = false; leftWidth = Math.max(minLeft, leftWidth) }}
          >Define projection</button>
        </div>
      {:else}
      <aside class="shrink-0" style={`width:${leftWidth}px;min-width:${minLeft}px ;max-width:${maxLeft}px;`}>
            <div class="tile tile-primary">
              <div class="tile-content" style={`height:${leftTileH}px`}>
                <div class="panel-actions"><button class="btn btn-xs btn-ui-secondary" title="Minimize" on:click={() => { leftCollapsed = true }}>–</button></div>
                <div class="tile-header mb-2 flex items-center gap-2"><span class="i-heroicons-photo text-slate-600" /> Define projection</div>
      <div class="relative" bind:this={minimapContainerRef} style={`width:100%;height:100%;`}>
        
        <div class="flex items-start">
          <div class="w-100 shrink-0">
            <AxesPanel
              {axes}
              embedSelection={embedSelection}
              projections={projectionOptions}
              customProjections={customProjections}
              on:embedChange={(e) => {
                const sel = e.detail.selection
                // If selecting a custom projection, assign its axes and do not reload gallery
                const cp = (customProjections || []).find(p => p.id === sel)
                if (cp) {
                  embedSelection = cp.id
                  if (cp.xAxisId) selectedAxisX = cp.xAxisId
                  if (cp.yAxisId) selectedAxisY = cp.yAxisId
                } else if (sel === 'meta') {
                  embedSelection = 'meta'
                  // pick first two metadata axes if available
                  const underGroup = axes.filter(a => (a?.group === 'meta') || idMatchesProjection('meta', a?.id || ''))
                  if (underGroup.length >= 2) {
                    selectedAxisX = underGroup[0].id
                    selectedAxisY = underGroup[1].id
                    logAxisDebug(selectedAxisX, 'Meta X')
                    logAxisDebug(selectedAxisY, 'Meta Y')
                  }
                } else {
                  embedSelection = sel
                  onEmbedChange()
                }
              }}
              on:setX={(e) => { selectedAxisX = e.detail.id; logAxisDebug(selectedAxisX, 'X') }}
              on:setY={(e) => { selectedAxisY = e.detail.id; logAxisDebug(selectedAxisY, 'Y') }}
              on:create={(e) => {
                const ax = e.detail
                if (ax && ax.id) {
                  // Ensure axis is grouped to current projection (not variant)
                  const groupKey = (embedSelection && String(embedSelection).startsWith('custom:')) ? embedSelection : (embedSelection || 'color_rgb')
                  const withGroup = { ...ax, group: ax.group || groupKey }
                  axes = [withGroup, ...axes]
                }
              }}
              on:delete={(e) => { axes = axes.filter(a => a.id !== e.detail.id) }}
              on:rename={(e) => { axes = axes.map(a => a.id === e.detail.id ? { ...a, name: e.detail.name } : a) }}
              on:addCustomProjection={(e) => {
                const name = (e.detail?.name || '').trim()
                if (!name) return
                const id = `custom:${Date.now()}`
                const xAxisId = selectedAxisX
                const yAxisId = selectedAxisY
                customProjections = [{ id, name, xAxisId, yAxisId }, ...customProjections]
                // Immediately select this custom projection and assign axes
                embedSelection = id
                if (xAxisId) selectedAxisX = xAxisId
                if (yAxisId) selectedAxisY = yAxisId
              }}
            />
            <Callout title="Define projections">
              <ul class="list-disc list-inside">
                <li> Start with an initial projection, </li>
                <li> Define new axes, and drag them to X/Y</li>
                <li> Save projection </li>
              </ul>
            </Callout>
          </div>
          <div class="flex-1 min-w-0 ml-2">
          <AxesMinimap
            items={allImages.map(i => ({ id: i.id, url: i.url, gx: i.gx, gy: i.gy, x: i.x, y: i.y }))}
            selections={selections}
            axes={axes}
            width={minimapW}
            height={minimapH}
            labels={new Map(Object.entries(labelDB))}
            bind:selectedX={selectedAxisX}
            bind:selectedY={selectedAxisY}
            on:axesChange={(e)=>{ selectedAxisX = e.detail.selectedX; selectedAxisY = e.detail.selectedY; console.log('[frontend] axesChange X/Y', selectedAxisX, selectedAxisY); logAxisDebug(selectedAxisX, 'X'); logAxisDebug(selectedAxisY, 'Y') }}
            on:label={onScribbleLabel}
            on:create={(e) => {
              const ax = e.detail
              if (ax && ax.id) {
                const groupKey = (embedSelection && String(embedSelection).startsWith('custom:')) ? embedSelection : (embedSelection || 'color_rgb')
                const withGroup = { ...ax, group: ax.group || groupKey }
                axes = [withGroup, ...axes]
              }
            }}
            on:saveSelection={(e) => { const sel = e.detail; if (sel && sel.id) { selections = [sel, ...selections] } }}
          />
          </div>
          
        </div>
        
      </div>
      <div class="tile-resize-handle" title="Resize" on:mousedown={startLeftTileResize}></div>
          </div>
        </aside>
      {/if}
     {#if middleCollapsed}
        <div class="shrink-0 pr-2" style="width:50px;">
          <button
            class="collapsed-handle rotate tile-header"
            title="Define Selections"
            on:click={() => { middleCollapsed = false }}
          >Define selections</button>
        </div>
      {:else}
        <section class="shrink-0 pl-2 pr-2" style={`width:${middleWidth}px`}>
          <div class="tile tile-primary">
            <div class="tile-content" style={`height:${middleTileH}px`}>
              <div class="panel-actions"><button class="btn btn-xs btn-ui-secondary" title="Minimize" on:click={() => { middleCollapsed = true }}>–</button></div>
              <div class="tile-header mb-2 flex items-center gap-2"><span class="i-heroicons-adjustments-horizontal text-slate-600" /> Define selections</div>
              
          <SelectionComposer
            {selections}
            items={allImages}
            on:toggle={(e)=>{ const { id, active } = e.detail; selections = selections.map(s => s.id===id ? { ...s, active: !!active } : s) }}
            on:rename={(e)=>{ const { id, name } = e.detail; if (!name) return; selections = selections.map(s => s.id===id ? { ...s, name: name.trim() } : s) }}
            on:delete={(e)=>{ const { id } = e.detail; selections = selections.filter(s => s.id !== id) }}
            on:combine={(e)=>{ const comb = e.detail; if (comb && comb.id) { combinedSelections = [comb, ...combinedSelections] } }}
          />
          <Callout title="Combine selections">
                Activate selections to combine all positives or discard all negatives.
              </Callout>
              <div class="tile-resize-handle" title="Resize" on:mousedown={startMiddleTileResize}></div>
            </div>
          </div>
                  <!-- Rightmost: saved combined selections -->
        {#if true}
          {#if rightCollapsed}
            <div class="shrink-0" style="width:50px;">
              <button
                class="collapsed-handle rotate tile-header"
                title="Show Combined selections"
                on:click={() => { rightCollapsed = false }}
              >Combined selections</button>
            </div>
          {:else}
            <aside class="shrink-0 mt-2" style={`width:${rightWidth}px;min-width:${minRight}px`}>
            <div class="tile tile-primary">
              <div class="tile-content" style={`height:${rightTileH}px`}>
                <div class="panel-actions"><button class="btn btn-xs btn-ui-secondary" title="Minimize" on:click={() => { rightCollapsed = true }}>–</button></div>
                <div class="tile-header mb-2 flex items-center gap-2"><span class="i-heroicons-rectangle-stack text-slate-600" /> Combined selections</div>
                  <div class="grid gap-2">
                    {#each combinedSelections as cs (cs.id)}
                      <CombinedSelection
                        combined={cs}
                        idToUrl={new Map(allImages.map(i => [i.id, i.url]))}
                        on:apply={(e)=>{
                          const { posIds=[], negIds=[] } = e.detail || {}
                          const next = {}
                          for (const id of allImages.map(i => i.id)) next[id] = undefined
                          for (const id of posIds) next[id] = 'pos'
                          for (const id of negIds) next[id] = 'neg'
                          labelDB = next
                        }}
                        on:rename={(e)=>{ const { id, name } = e.detail; if (!name) return; combinedSelections = combinedSelections.map(s => s.id===id ? { ...s, name: name.trim() } : s) }}
                        on:delete={(e)=>{ const { id } = e.detail; combinedSelections = combinedSelections.filter(s => s.id !== id) }}
                        on:sendToSelections={(e)=>{ const { selection } = e.detail || {}; if (selection && selection.id) { selections = [selection, ...selections] } }}
                      />
                    {/each}
                    {#if combinedSelections.length === 0}
                      <div class="text-sm text-gray-500">No combined selections yet. Use the Selection composer to create one.</div>
                    {/if}
                  </div>
                  <Callout title="Saved">
                    Apply a combined selection to set current labels, or move to selections to edit.
                  </Callout>
                  <div class="tile-resize-handle" title="Resize" on:mousedown={startRightTileResize}></div>
                </div>
              </div>
            </aside>
          {/if}
        {/if}

        </section>
      {/if}

      

    </div>
  </main>
</div>

<style>
  :global(html, body, #app) { height: 100%; }
</style>
