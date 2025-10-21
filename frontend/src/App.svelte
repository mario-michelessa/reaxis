<script>
  // import ImageGrid from './components/ImageGrid.svelte' // replaced by TextSimilarity view
  import { generateDemoImages, sortBySimilarity, clusterKMeans } from './lib/data'
  import { onMount, tick } from 'svelte'
  import ScatterMinimap from './components/ScatterMinimap.svelte'
  import AnimatedMinimap from './components/AnimatedMinimap.svelte'
  import HoverGridMinimap from './components/HoverGridMinimap.svelte'
  import AxesMinimap from './components/AxesMinimap.svelte'
  import AxesPanel from './components/AxesPanel.svelte'
  import AxisBuilder from './components/AxisBuilder.svelte'
  import DiftPartSelector from './components/DiftPartSelector.svelte'
  import MinimapTextOverlay from './components/MinimapTextOverlay.svelte'
  import ConceptsPanel from './components/ConceptsPanel.svelte'
  import TextSimilarity from './components/TextSimilarity.svelte'
  import ConceptComposer from './components/ConceptComposer.svelte'
  import CombinedConceptItem from './components/CombinedConceptItem.svelte'
  
  let allImages = []
  let prevImages = []
  // Cache of gallery responses by key (dataset|embed|method)
  const galleryCache = new Map()
  let classes = []
  let classFilter = 'All'
  let viewMode = 'gallery' // 'gallery' | 'cluster' | 'similarity'
  let selected = new Set()
  let refId = null

  let filtered = []
  let clusters = []
  
  // Backend wiring
  // const API_BASE = (import.meta.env && import.meta.env.VITE_API_BASE) ? import.meta.env.VITE_API_BASE : 'http://127.0.0.1:5001'
  const API_BASE = (import.meta.env && import.meta.env.VITE_API_BASE) ? import.meta.env.VITE_API_BASE : 'http://localhost:5002'
  console.log('[frontend] API_BASE', API_BASE)
  let datasetPath = '' // leave blank to let server pick sample
  // Radio selection and effective embed method
  let embedSelection = 'avg' // 'avg' | 'clip' | 'dino' | 'dift_sd' | 'text'
  let embedMethod = 'avg' // effective method sent to backend (may be dift_sd_partXY)
  let gridSize = 0
  let warningMsg = ''
  // External label storage (object, separate from images)
  let labelDB = {}
  let labelsMap = new Map()
  $: goodCount = Object.values(labelDB).filter(v => v === 'good').length
  $: badCount = Object.values(labelDB).filter(v => v === 'bad').length

  // Scribble controls
  let scribbleEnabled = false
  let scribbleLabel = 'good' // 'good' | 'bad'
  let scribbleRadius = 18
  let minimapRef
  let textOverlayRef
  let minimapContainerRef
  // Dynamic minimap size based on available space
  $: minimapSize = (() => {
    // Recompute when left pane or window resizes
    void leftWidth; void windowWidth
    const w = minimapContainerRef ? Math.floor(minimapContainerRef.clientWidth - 210 || 700) : 700
    return Math.max(300, Math.min(1200, w))
  })()
  let showTextOverlay = false
  let textSimilarities = {}
  // Text-driven separation layer state (persists)
  let textLayerState = { baseEmbed: '', baseCoords: {}, coords: {}, rects: [], gridSize: 0 }

  // Axes state
  let axes = [] // [{ id, name, coords }]
  let selectedAxisX = null
  let selectedAxisY = null

  function scatterToCenter(radius = 0.06) {
    // Randomly scatter all points around center within a small radius
    prevImages = allImages
    const r = Math.max(0.005, Math.min(0.2, Number(radius) || 0.06))
    allImages = allImages.map((it) => {
      const ang = Math.random() * Math.PI * 2
      // sqrt for uniform density in circle
      const rad = Math.sqrt(Math.random()) * r
      const nx = Math.max(0, Math.min(1, 0.5 + Math.cos(ang) * rad))
      const ny = Math.max(0, Math.min(1, 0.5 + Math.sin(ang) * rad))
      return { ...it, x: nx, y: ny }
    })
    // Persist scattered coords into text layer state
    const next = {}
    for (const it of allImages) next[it.id] = [it.x, it.y]
    textLayerState = { ...textLayerState, coords: next }
  }

  function ensureTextBase() {
    if (Object.keys(textLayerState.baseCoords || {}).length > 0) return
    const baseCoords = {}
    for (const it of allImages) baseCoords[it.id] = [it.x, it.y]
    textLayerState = { ...textLayerState, baseCoords, ids: allImages.map(it => it.id) }
  }

  function applyTextCoordsFromState() {
    if (!textLayerState || !textLayerState.coords) return
    const map = textLayerState.coords
    prevImages = allImages
    allImages = allImages.map((it) => {
      const c = map[it.id]
      return c ? { ...it, x: Number(c[0]), y: Number(c[1]) } : it
    })
  }
  // Split pane state
  let leftWidth = 1500
  let dragging = false
  let startX = 0
  let startLeft = 0
  let lastMouseX = 0
  const minLeft = 740 // minimum visible width for the left pane
  const maxLeft = 1400
  const collapseThreshold = 700 // drag below this to auto-collapse on release
  let leftCollapsed = false
  function startDrag(e) { dragging = true; startX = e.clientX; startLeft = leftWidth; lastMouseX = e.clientX }
  function onLeftResizerDblClick() {
    if (leftCollapsed) {
      leftCollapsed = false
      leftWidth = Math.max(minLeft, leftWidth)
    } else {
      leftCollapsed = true
    }
  }
  
  // Right pane resizer (between composer and combined concepts)
  let rightWidth = 320 // default ~w-80
  let draggingRight = false
  let startXRight = 0
  let startRight = 0
  const minRight = 180
  function startRightDrag(e) { draggingRight = true; startXRight = e.clientX; startRight = rightWidth }
  function onRightResizerDblClick() {
    middleCollapsed = !middleCollapsed
  }
  
  let windowWidth = 0
  let rowRef
  let rowWidth = 0
  $: rowWidth = rowRef ? rowRef.clientWidth : windowWidth
  let middleCollapsed = false
  function onDrag(e) {
    lastMouseX = e.clientX
    if (dragging) {
      const dx = e.clientX - startX
      // While dragging, keep within visible bounds; actual collapse applied on mouseup
      leftWidth = Math.max(minLeft, Math.min(maxLeft, startLeft + dx))
    }
    if (draggingRight) {
      const dxr = startXRight - e.clientX // moving left increases right pane width
      const tentative = startRight + dxr
      const RESIZER_PX = 4
      const leftVisible = leftCollapsed ? 28 : leftWidth
      const minMiddle = 200 // enforce minimum middle width while dragging
      const total = rowWidth || windowWidth || 0
      const allowedMaxRight = Math.max(minRight, total - leftVisible - minMiddle - (2 * RESIZER_PX))
      rightWidth = Math.max(minRight, Math.min(tentative, allowedMaxRight))
    }
  }
  function endDrag() {
    if (dragging) {
      const tentative = startLeft + (lastMouseX - startX)
      if (!leftCollapsed && tentative < collapseThreshold) {
        leftCollapsed = true
        // Reset to a sensible width for when expanded next time
        leftWidth = Math.max(minLeft, leftWidth)
      }
    }
    if (draggingRight) {
      // Estimate middle width and collapse if too small
      const RESIZER_PX = 4
      const leftVisible = leftCollapsed ? 28 : leftWidth
      const total = rowWidth || windowWidth || 0
      const estimatedMiddle = Math.max(0, total - leftVisible - rightWidth - (2 * RESIZER_PX))
      const middleCollapseThreshold = 300
      if (estimatedMiddle < middleCollapseThreshold) {
        middleCollapsed = true
      }
    }
    dragging = false; draggingRight = false
  }

  // DIFT part selector (for UI display only)
  let diftPart = '' // e.g., '11', '12', ..., '33'

  // UI aliases for embed methods
  function methodAlias(m) {
    if (!m) return ''
    if (m === 'avg') return 'Color'
    if (m === 'clip') return 'Content'
    if (m === 'dino') return 'Global composition'
    if (m === 'dift_sd') return 'Local composition'
    if (m === 'text') return 'Text'
    if (m.startsWith('dift_sd_part')) return `Local composition ${m.replace('dift_sd_part','part ')}`
    return m
  }

  // Create or update the two default axes for the current projection (x,y)
  function axisPrefixForMethod(method) {
    if (!method) return 'Axis'
    if (method === 'avg') return 'Color'
    if (method === 'clip') return 'Content'
    if (method === 'dino') return 'Global composition'
    if (method.startsWith('dift_sd_part')) return `Local composition ${method.replace('dift_sd_part','part ')}`
    if (method === 'dift_sd') return 'Local composition'
    return method
  }

  function ensureDefaultAxesForCurrentProjection() {
    const methodName = embedMethod || currentMethodLabel()
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
    let next = axes
    const axX = { id: idX, name: `${prefix} axis 1`, coords: coordsX }
    const axY = { id: idY, name: `${prefix} axis 2`, coords: coordsY }
    const hasX = next.some(a => a.id === idX)
    const hasY = next.some(a => a.id === idY)
    if (hasX) next = next.map(a => a.id === idX ? axX : a); else next = [axX, ...next]
    if (hasY) next = next.map(a => a.id === idY ? axY : a); else next = [axY, ...next]
    axes = next
    // Default selection to current projection axes
    selectedAxisX = idX
    selectedAxisY = idY
  }

  // Concepts: saved sets of labels tied to an embedding method
  // Shape: { id, name, method, good: string[], bad: string[] }
  let concepts = []
  // Saved combined concepts for the right panel
  let combinedConcepts = []
  let composerLoadChain = null

  function currentMethodLabel() {
    // In Text mode, reflect the base embedding, not 'text'
    if (embedSelection === 'text') {
      return (textLayerState.baseEmbed && String(textLayerState.baseEmbed)) || (embedMethod || 'clip')
    }
    // Prefer effective embedMethod which includes dift part when applicable
    return embedMethod || (embedSelection === 'dift_sd' && diftPart ? `dift_sd_part${diftPart}` : embedSelection)
  }

  function createConcept() {
    const methodName = currentMethodLabel()
    if (!methodName) {
      alert('Select an embedding/method before creating a concept.')
      return
    }
    const proposed = prompt('Name this concept', methodName)
    if (!proposed || !proposed.trim()) {
      // User cancelled or provided empty name; do not create
      return
    }
    const good = Object.entries(labelDB).filter(([, v]) => v === 'good').map(([k]) => k)
    const bad = Object.entries(labelDB).filter(([, v]) => v === 'bad').map(([k]) => k)
    const id = `${methodName}:${Date.now()}`
    const concept = { id, name: proposed.trim(), method: methodName, good, bad }
    concepts = [concept, ...concepts]
    // flush current labels
    labelsMap = new Map()
    labelDB = {}
  }

  function createConceptFromGood(goodIds) {
    const methodName = currentMethodLabel()
    if (!methodName) {
      alert('Select an embedding/method before creating a concept.')
      return
    }
    const id = `${methodName}:${Date.now()}`
    const concept = { id, name: methodName, method: methodName, good: [...goodIds], bad: [] }
    concepts = [concept, ...concepts]
  }

  async function loadConcept(concept) {
    // Switch embedding selection to the concept's method and reload gallery
    const m = concept.method || concept.name
    if (m && m.startsWith('dift_sd_part')) {
      diftPart = m.replace('dift_sd_part', '')
      embedSelection = 'dift_sd'
      embedMethod = `dift_sd_part${diftPart}`
    } else {
      embedSelection = m
      embedMethod = m
    }
    await loadGallery()
    // Restore labels
    const nextMap = new Map()
    const nextDB = {}
    for (const id of concept.good || []) { nextMap.set(id, 'good'); nextDB[id] = 'good' }
    for (const id of concept.bad || []) { nextMap.set(id, 'bad'); nextDB[id] = 'bad' }
    labelsMap = nextMap
    labelDB = nextDB
  }

  function removeConcept(id) {
    concepts = concepts.filter(c => c.id !== id)
  }

  function renameConceptById(id, name) {
    if (!name || !name.trim()) return
    concepts = concepts.map(c => c.id === id ? { ...c, name: name.trim() } : c)
  }

  function addCombinedConcept(detail) {
    const name = (detail?.name || '').trim() || 'Combined'
    const methodName = (detail?.method || '').trim() || currentMethodLabel()
    const id = `${methodName}:${Date.now()}`
    const good = Array.isArray(detail?.good) ? detail.good : []
    const bad = Array.isArray(detail?.bad) ? detail.bad : []
    const chain = detail?.chain || null
    const concept = { id, name, method: methodName, good, bad, chain }
    combinedConcepts = [concept, ...combinedConcepts]
  }

  function applyFilters() {
    filtered = allImages.filter((i) => classFilter === 'All' || i.className === classFilter)
    if (viewMode === 'similarity' && refId) {
      filtered = sortBySimilarity(filtered, refId)
    }
    if (viewMode === 'cluster') {
      clusters = clusterKMeans(filtered, 3)
    } else {
      clusters = []
    }
  }

  function toggleSelect(id) {
    if (selected.has(id)) selected.delete(id)
    else selected.add(id)
    selected = new Set(selected)
  }

  function setReference(item) {
    refId = item.id
    applyFilters()
  }

  function clearSelection() {
    selected = new Set()
  }

  function selectAllCurrent() {
    const ids = (viewMode === 'cluster' ? clusters.flatMap((c) => c.items) : filtered).map((i) => i.id)
    selected = new Set(ids)
  }

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
      gridSize = Number(cached.n_layer || 0)
      classes = ['All', ...Array.from(new Set(allImages.map((i) => i.className)))]
      applyFilters()
      // Populate axes from current projection (x,y)
      ensureDefaultAxesForCurrentProjection()
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
      gridSize = Number(data.n_layer || 0)
      classes = ['All', ...Array.from(new Set(allImages.map((i) => i.className)))]
      // Cache for quick toggling between embeddings
      galleryCache.set(cacheKey, { items: allImages, n_layer: gridSize })
      applyFilters()
      // Populate axes from current projection (x,y)
      ensureDefaultAxesForCurrentProjection()
    } catch (e) {
      console.error('[frontend] fetch error', e)
      // Fallback to demo data
      warningMsg = ''
      allImages = generateDemoImages(48)
      classes = ['All', ...Array.from(new Set(allImages.map((i) => i.className)))]
      applyFilters()
    }
  }

  onMount(() => {
    try {
      const raw = localStorage.getItem('promptherder.concepts')
      if (raw) {
        const parsed = JSON.parse(raw)
        if (Array.isArray(parsed)) concepts = parsed
      }
      const rawComb = localStorage.getItem('promptherder.combined')
      if (rawComb) {
        const parsedC = JSON.parse(rawComb)
        if (Array.isArray(parsedC)) combinedConcepts = parsedC
      }
      const rawText = localStorage.getItem('promptherder.textlayer')
      if (rawText) {
        const parsedT = JSON.parse(rawText)
        if (parsedT && typeof parsedT === 'object') textLayerState = parsedT
      }
      const rawAxes = localStorage.getItem('promptherder.axes')
      if (rawAxes) {
        const parsedA = JSON.parse(rawAxes)
        if (Array.isArray(parsedA)) axes = parsedA
      }
    } catch (e) { /* ignore */ }
    loadGallery()
  })

  $: (function persistConcepts(c) {
    try { localStorage.setItem('promptherder.concepts', JSON.stringify(c)) } catch (_) {}
  })(concepts)
  $: (function persistCombined(c) {
    try { localStorage.setItem('promptherder.combined', JSON.stringify(c)) } catch (_) {}
  })(combinedConcepts)

  $: (function persistTextLayer(s) {
    try { localStorage.setItem('promptherder.textlayer', JSON.stringify(s)) } catch (_) {}
  })(textLayerState)
  $: (function persistAxes(a) {
    try { localStorage.setItem('promptherder.axes', JSON.stringify(a)) } catch (_) {}
  })(axes)

  function onEmbedChange() {
    // Compute effective method; if DIFT selected without part, don't fetch yet
    if (embedSelection === 'dift_sd' && !diftPart) {
      warningMsg = 'Select a DIFT part (3x3) to load embeddings'
      embedMethod = ''
      return
    }
    warningMsg = ''
    // Do not set embedMethod to 'text'; preserve last non-text embedding
    if (embedSelection === 'dift_sd') {
      embedMethod = `dift_sd_part${diftPart}`
    } else if (embedSelection !== 'text') {
      embedMethod = embedSelection
    }
    if (embedSelection === 'text') {
      // Enter text layer: disable scribble, set base if missing, and apply/preset coords
      scribbleEnabled = false
      if (!(textLayerState && Object.keys(textLayerState.baseCoords||{}).length > 0)) {
        scatterToCenter(0.06)
        // After scatter, set the base to the scattered positions
        ensureTextBase()
      } else if (textLayerState && Object.keys(textLayerState.coords||{}).length > 0) {
        applyTextCoordsFromState()
      }
    } else {
      // Switching to a normal embed; reload gallery
      loadGallery()
    }
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
        else if (v === 'good' || v === 'bad') next[id] = v
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

  function clearAllLabels() {
    // Clear overlay and labels
    if (minimapRef && minimapRef.clearScribble) minimapRef.clearScribble()
    labelsMap = new Map()
    labelDB = {}
  }

  function onConfirmRegion(e) {
    const { rect, text } = e.detail
    // Update saved rectangles for persistence
    const nextRects = Array.isArray(textLayerState.rects) ? [...textLayerState.rects] : []
    nextRects.push({ ...rect, text })
    // Build ids and base coords array based on initial scattered base
    const ids = (textLayerState.ids && Array.isArray(textLayerState.ids) && textLayerState.ids.length === allImages.length)
      ? textLayerState.ids
      : allImages.map(it => it.id)
    const baseCoordsArr = ids.map(id => {
      const bc = textLayerState.baseCoords?.[id]
      return Array.isArray(bc) ? [Number(bc[0]), Number(bc[1])] : [0.5, 0.5]
    })
    fetch(`${API_BASE}/text_forces`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        texts: nextRects.map(r => ({ text: r.text, rect: { x: r.x, y: r.y, w: r.w, h: r.h } })),
        ids,
        base_coords: baseCoordsArr,
        embed: 'clip',
        method: 'pca',
        alpha: 0.35
      })
    }).then(async (res) => {
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      const coords = data.coords || []
      const packed = data.packed || []
      const nlayer = Number(data.n_layer || gridSize)
      const idOrder = Array.isArray(data.ids) ? data.ids : ids
      // Update coords map according to returned order
      const nextMap = {}
      for (let i = 0; i < idOrder.length; i++) {
        const id = idOrder[i]
        const c = coords[i]
        if (Array.isArray(c) && c.length >= 2) nextMap[id] = [Number(c[0]), Number(c[1])]
      }
      textLayerState = { ...textLayerState, rects: nextRects, coords: nextMap }
      // Apply to visible items
      prevImages = allImages
      const posById = new Map(Object.entries(nextMap))
      allImages = allImages.map((it, idx) => ({
        ...it,
        x: posById.has(it.id) ? Number(posById.get(it.id)[0]) : it.x,
        y: posById.has(it.id) ? Number(posById.get(it.id)[1]) : it.y,
        gx: packed[idx] ? packed[idx][0] : it.gx,
        gy: packed[idx] ? packed[idx][1] : it.gy,
      }))
      gridSize = nlayer
    }).catch((err) => {
      console.error('text_forces error', err)
    })
  }
</script>

<svelte:window bind:innerWidth={windowWidth} on:mousemove={onDrag} on:mouseup={endDrag} />

<div class="min-h-screen bg-white text-gray-900">
  <header class="sticky top-0 z-10">
    <div class="w-full bg-gradient-to-r from-slate-100 via-slate-200 to-slate-100 border-b border-slate-200">
      <div class="px-4 py-3 flex items-center gap-3">
        <div class="i-heroicons-photo inline-block text-slate-600 text-2xl" />
        <div class="text-lg font-semibold tracking-wide text-slate-800">PromptHerder</div>
        <div class="flex-1" />
      </div>
    </div>
  </header>

  {#if warningMsg}
    <div class="bg-yellow-50 border-l-4 border-yellow-400 text-yellow-800 p-3">
      <div class="container mx-auto px-4 text-sm">
        {warningMsg} — run: <code>python backend/precompute_embeddings.py {datasetPath || '[DATASET_PATH]'} --methods avg,clip,dino,dift_sd</code>
      </div>
    </div>
  {/if}

  <main class="w-full py-6">
    <div class="flex gap-0 flex-nowrap overflow-x-auto" bind:this={rowRef}>
      <!-- Left minimap + list -->
      {#if leftCollapsed}
        <div class="shrink-0" style="width:30px; height:200px">
          <button
            class="w-full h-full flex items-center justify-center text-sm font-semibold mb-2 text-slate-700 border border-slate-200 bg-gradient-to-b from-slate-50 to-slate-200 hover:from-slate-100 hover:to-slate-300"
            style="writing-mode: vertical-lr; text-orientation: sideways; transform: rotate(180deg);"
            title="Show Images gallery visualization"
            on:click={() => { leftCollapsed = false; leftWidth = Math.max(minLeft, leftWidth) }}
          >Images gallery visualization</button>
        </div>
      {:else}
      <aside class="shrink-0 pr-4" style={`width:${leftWidth}px;min-width:${minLeft}px`}>
          <div class="bg-gradient-to-r from-slate-100 via-slate-200 to-slate-100 text-slate-800 px-3 py-2 rounded-md border border-slate-200 text-sm font-semibold mb-2 flex items-center gap-2">
            <span class="i-heroicons-photo text-slate-600" />
            <span>Images gallery visualization</span>
          </div>
      <!-- Removed legacy scribble label toggle; lasso handles labeling inside minimap -->
      {#if embedSelection === 'dift_sd'}
        <!-- DIFT part selector moved to AxesPanel -->
      {/if}
      <div class="relative" bind:this={minimapContainerRef} style={`width:100%;height:${minimapSize}px;`}>
        <!-- Method selection moved to AxesPanel -->
        {#if embedSelection === 'text'}
          <div class="absolute top-1 right-1 z-10 bg-white/90 rounded shadow px-2 py-1 text-xs flex items-center gap-2">
            <button class="px-2 py-1 border rounded inline-flex items-center gap-1" on:click={async () => { showTextOverlay = true; await tick(); if (textOverlayRef && textOverlayRef.startPlacing) textOverlayRef.startPlacing() }}>
              <span class="i-heroicons-rectangle-group" /> Add text
            </button>
            <button class="px-2 py-1 border rounded inline-flex items-center gap-1" on:click={() => { scatterToCenter(0.06) }}>
              <span class="i-heroicons-arrow-path" /> Reset
            </button>
          </div>
        {/if}
        <div class="flex items-start gap-3">
          <div class="w-56 shrink-0">
            <div class="text-xs text-gray-700 mb-1">Axes</div>
            <AxesPanel
              {axes}
              items={allImages}
              {concepts}
              embedSelection={embedSelection}
              diftPart={diftPart}
              on:embedChange={(e) => { embedSelection = e.detail.selection; onEmbedChange() }}
              on:selectDiftPart={(e) => { diftPart = e.detail.part; embedMethod = `dift_sd_part${diftPart}`; embedSelection = 'dift_sd'; loadGallery(); }}
              on:setX={(e) => { selectedAxisX = e.detail.id }}
              on:setY={(e) => { selectedAxisY = e.detail.id }}
              on:create={(e) => { const ax = e.detail; if (ax && ax.id) { axes = [ax, ...axes] } }}
              on:delete={(e) => { axes = axes.filter(a => a.id !== e.detail.id) }}
              on:rename={(e) => { axes = axes.map(a => a.id === e.detail.id ? { ...a, name: e.detail.name } : a) }}
            />
          </div>
          <div class="flex-1 min-w-0">
            <AxesMinimap
              items={allImages.map(i => ({ id: i.id, url: i.url, gx: i.gx, gy: i.gy, x: i.x, y: i.y }))}
              axes={axes}
              width={minimapSize}
              height={minimapSize}
              labels={new Map(Object.entries(labelDB))}
              bind:selectedX={selectedAxisX}
              bind:selectedY={selectedAxisY}
              on:axesChange={(e)=>{ selectedAxisX = e.detail.selectedX; selectedAxisY = e.detail.selectedY }}
              on:label={onScribbleLabel}
              on:create={(e) => { const ax = e.detail; if (ax && ax.id) { axes = [ax, ...axes] } }}
            />
          </div>
        </div>
        {#if embedSelection === 'text' && showTextOverlay}
          <MinimapTextOverlay
            bind:this={textOverlayRef}
            width={minimapSize}
            height={minimapSize}
            rectangles={textLayerState.rects}
            on:confirmRegion={onConfirmRegion}
          />
        {/if}
      </div>
      

      <div class="text-sm mb-1 mt-4">Concepts list</div>
      <ConceptsPanel
        {concepts}
        items={allImages}
        on:select={(e) => loadConcept(e.detail.concept)}
        on:delete={(e) => removeConcept(e.detail.id)}
        on:rename={(e) => renameConceptById(e.detail.id, e.detail.name)}
      />

        </aside>
      {/if}
<!-- 
    <section class="flex-1">
      <div class="text-sm mb-1">Search images</div>
      <TextSimilarity
        items={allImages}
        apiBase={API_BASE}
        defaultTopN={10}
        onCreateConcept={(ids) => createConceptFromGood(ids)}
        onRefreshGallery={async () => { await loadGallery() }}
      />
    </section> 
-->

      <!-- Resizer -->
      <div class="w-1 bg-gray-200 hover:bg-gray-300 cursor-col-resize" title="Resize left panel (double-click to collapse/expand)" on:mousedown={startDrag} on:dblclick={onLeftResizerDblClick} />

      <!-- Middle: concepts composer -->
      {#if middleCollapsed}
        <div class="shrink-0" style="width:30px; height:200px">
          <button
            class="w-full h-full flex items-center justify-center text-[11px] text-slate-700 border border-slate-200 bg-gradient-to-b from-slate-50 to-slate-200 hover:from-slate-100 hover:to-slate-300"
            style="writing-mode: vertical-lr; text-orientation: upright;"
            title="Show Concept composer"
            on:click={() => { middleCollapsed = false }}
          >Concept composer</button>
        </div>
      {:else}
        <section class="flex-1 min-w-0 pl-4 pr-4">
          <div class="bg-gradient-to-r from-slate-100 via-slate-200 to-slate-100 text-slate-800 px-3 py-2 rounded-md border border-slate-200 text-sm font-semibold mb-2 flex items-center gap-2">
            <span class="i-heroicons-adjustments-horizontal text-slate-600" />
            <span>Concept composer</span>
          </div>
          <div class="mt-4">
            <AxisBuilder
              items={allImages}
              on:create={(e) => { const ax = e.detail; if (ax && ax.id) { axes = [ax, ...axes] } }}
            />
          </div>
          <ConceptComposer
            {concepts}
            allIds={allImages.map(i => i.id)}
            defaultMethod={currentMethodLabel()}
            items2d={allImages}
            apiBase={API_BASE}
            datasetPath={datasetPath}
            on:create={(e) => addCombinedConcept(e.detail)}
            loadChain={composerLoadChain}
          />
        </section>
      {/if}

      <!-- Resizer between composer and right panel -->
      <div class="w-1 bg-gray-200 hover:bg-gray-300 cursor-col-resize" title="Resize right panel (double-click to collapse/expand middle)" on:mousedown={startRightDrag} on:dblclick={onRightResizerDblClick} />

      <!-- Rightmost: saved combined concepts -->
      <aside class="shrink-0" style={`width:${rightWidth}px;min-width:${minRight}px`}>
        <div class="bg-gradient-to-r from-slate-100 via-slate-200 to-slate-100 text-slate-800 px-3 py-2 rounded-md border border-slate-200 text-sm font-semibold mb-2 flex items-center gap-2">
          <span class="i-heroicons-rectangle-stack text-slate-600" />
          <span>Saved combined concepts</span>
        </div>
        <div class="grid gap-2">
          {#each combinedConcepts as cc (cc.id)}
            <CombinedConceptItem
              combined={cc}
              idToUrl={new Map(allImages.map(i => [i.id, i.url]))}
              allIds={allImages.map(i => i.id)}
              on:load={(e) => { composerLoadChain = e.detail.chain || null }}
            />
          {/each}
          {#if combinedConcepts.length === 0}
            <div class="text-xs text-gray-500">No combined concepts yet. Use the composer to create one.</div>
          {/if}
        </div>
      </aside>
    </div>
  </main>
</div>

<style>
  :global(html, body, #app) { height: 100%; }
</style>
