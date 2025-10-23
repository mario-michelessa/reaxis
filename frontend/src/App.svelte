<script>
  // import ImageGrid from './components/ImageGrid.svelte' // replaced by TextSimilarity view
  import { generateDemoImages, sortBySimilarity, clusterKMeans } from './lib/data'
  import { onMount, tick } from 'svelte'
  import AxesMinimap from './components/AxesMinimap.svelte'
  import AxesPanel from './components/AxesPanel.svelte'
  import MinimapTextOverlay from './components/MinimapTextOverlay.svelte'
  import ConceptsPanel from './components/ConceptsPanel.svelte'
  import TextSimilarity from './components/TextSimilarity.svelte'
  import ConceptComposer from './components/ConceptComposer.svelte'
  import CombinedConceptItem from './components/CombinedConceptItem.svelte'
  import Callout from './components/Callout.svelte'
  import CombinedSelection from './components/CombinedSelection.svelte'
  import SelectionComposer from './components/SelectionComposer.svelte'
  
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
  // Datasets list for header dropdown; populated from backend if available
  let datasets = [
    { label: 'Sample (server default)', value: '' },
    { label: 'COCO (sample)', value: 'coco-sample' },
    { label: 'CIFAR-10', value: 'cifar10' },
    { label: 'Placeholder A', value: 'dataset-a' },
  ]
  // Radio selection and effective embed method
  let embedSelection = 'avg' // 'avg' | 'clip' | 'dino' | 'dift_sd' | 'text'
  let embedMethod = 'avg' // effective method sent to backend (may be dift_sd_partXY)
  let colorSpace = 'color_lch' // 'color_lch' | 'color_hsv'
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
    void leftWidth; void windowWidth; void leftTileH;
    const w = minimapContainerRef ? Math.floor(minimapContainerRef.clientWidth - 330) : 700
    const h = minimapContainerRef ? Math.floor(minimapContainerRef.clientHeight - 50) : 700
    return Math.max(300, Math.min(2000, w, h))
  })()
  let showTextOverlay = false
  let textSimilarities = {}
  // Text-driven separation layer state (persists)
  let textLayerState = { baseEmbed: '', baseCoords: {}, coords: {}, rects: [], gridSize: 0 }

  // Axes state
  let axes = [] // [{ id, name, coords }]
  let selectedAxisX = null
  let selectedAxisY = null
  
  function resetAllStateForDatasetChange() {
    // Clear in-memory state
    allImages = []
    prevImages = []
    filtered = []
    clusters = []
    selected = new Set()
    refId = null
    axes = []
    selectedAxisX = null
    selectedAxisY = null
    concepts = []
    combinedConcepts = []
    selections = []
    combinedSelections = []
    labelsMap = new Map()
    labelDB = {}
    textLayerState = { baseEmbed: '', baseCoords: {}, coords: {}, rects: [], gridSize: 0 }
    showTextOverlay = false
    warningMsg = ''
    // Clear caches
    galleryCache.clear()
    // Clear persisted keys
    try {
      localStorage.removeItem('promptherder.concepts')
      localStorage.removeItem('promptherder.combined')
      localStorage.removeItem('promptherder.textlayer')
      localStorage.removeItem('promptherder.axes')
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
  let leftWidth = 1200
  // Simple per-tile sizes for corner resize
  let leftTileH = 920
  let middleWidth = 700
  let middleTileH = 600
  let rightTileH = 600
  let dragging = false
  let startX = 0
  let startLeft = 0
  let lastMouseX = 0
  let minLeft = 500 // tune: minimum visible width for the left pane
  let maxLeft = 2000 // tune: maximum visible width for the left pane
  let collapseThreshold = 700 // tune: drag below this to auto-collapse on release
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
  let rightCollapsed = true
  let draggingRight = false
  let startXRight = 0
  let startRight = 0
  let minRight = 180 // tune: minimum right panel width
  function startRightDrag(e) { draggingRight = true; startXRight = e.clientX; startRight = rightWidth }
  function onRightResizerDblClick() {
    middleCollapsed = !middleCollapsed
  }
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
  let rowRef
  let rowWidth = 0
  $: rowWidth = rowRef ? rowRef.clientWidth : windowWidth
  let middleCollapsed = true
  // Projection options + aliases forwarded to AxesPanel
  // Combine Shape and Local similarity under one radio; default DINO, optional DIFT part
  const projectionOptions = [
    { value: 'avg', label: 'Color' },
    { value: 'clip', label: 'Semantic' },
    { value: 'shape', label: 'Shape' },
    { value: 'text', label: 'Text' },
  ]
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
      const minMiddle = 260 // tune: minimum middle width while dragging
      const total = rowWidth || windowWidth || 0
      const allowedMaxRight = Math.max(minRight, total - leftVisible - minMiddle - (2 * RESIZER_PX))
      rightWidth = Math.max(minRight, Math.min(tentative, allowedMaxRight))
    }
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
    tileDrag = null
    dragging = false; draggingRight = false
  }

  // DIFT part selector (for UI display only)
  let diftPart = '' // e.g., '11', '12', ..., '33'

  // Create or update the two default axes for the current projection (x,y)
  function axisPrefixForMethod(method) {
    if (!method) return 'Axis'
    if (method === 'avg') return 'Color'
    if (method === 'color_hsv') return 'Color HSV'
    if (method === 'color_lch') return 'Color LCh'
    if (method === 'clip') return 'Semantic'
    if (method === 'dino') return 'Shape'
    if (method.startsWith('dift_sd_part')) return `Local sim. ${method.replace('dift_sd_part','part ')}`
    if (method === 'dift_sd') return 'Local sim.'
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
    const axX = { id: idX, name: `${prefix} X`, coords: coordsX }
    const axY = { id: idY, name: `${prefix} Y`, coords: coordsY }
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
  // Selections state
  let selections = []
  let combinedSelections = []

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
  $: (function persistSelections(s) {
    try { localStorage.setItem('promptherder.selections', JSON.stringify(s)) } catch (_) {}
  })(selections)
  $: (function persistCombinedSelections(s) {
    try { localStorage.setItem('promptherder.combinedSelections', JSON.stringify(s)) } catch (_) {}
  })(combinedSelections)

  function onEmbedChange() {
    warningMsg = ''
    // Compute effective method for each selection
    if (embedSelection === 'shape') {
      embedMethod = diftPart ? `dift_sd_part${diftPart}` : 'dino'
    } else if (embedSelection === 'avg') {
      embedMethod = colorSpace || 'avg'
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
        {warningMsg} — run: <code>python backend/precompute_embeddings.py {datasetPath || '[DATASET_PATH]'} --methods avg,clip,dino,dift_sd</code>
      </div>
    </div>
  {/if}

  <main class="app-main w-full py-6">
    <div class="panels-row flex flex-nowrap overflow-x-auto" bind:this={rowRef}>
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
        {#if embedSelection === 'text'}
          <div class="absolute top-1 z-10 bg-white/90 rounded shadow px-2 py-1 text-sm flex items-center gap-2" style="left: 50%;">
            <button class="px-2 py-1 border rounded inline-flex items-center gap-1" on:click={async () => { showTextOverlay = true; await tick(); if (textOverlayRef && textOverlayRef.startPlacing) textOverlayRef.startPlacing() }}>
              <span class="i-heroicons-rectangle-group" /> Add text
            </button>
            <button class="px-2 py-1 border rounded inline-flex items-center gap-1" on:click={() => { scatterToCenter(0.06) }}>
              <span class="i-heroicons-arrow-path" /> Reset
            </button>
          </div>
        {/if}
        <div class="flex items-start">
          <div class="w-80 shrink-0">
            <AxesPanel
              {axes}
              items={allImages}
              {concepts}
              labels={new Map(Object.entries(labelDB))}
              embedSelection={embedSelection}
              diftPart={diftPart}
              colorSpace={colorSpace}
              projections={projectionOptions}
              on:embedChange={(e) => { embedSelection = e.detail.selection; onEmbedChange() }}
              on:selectDiftPart={(e) => {
                diftPart = (e.detail.part || '').trim()
                if (diftPart) {
                  embedMethod = `dift_sd_part${diftPart}`
                } else {
                  embedMethod = 'dino'
                }
                embedSelection = 'shape'
                loadGallery()
              }}
              on:selectColorSpace={(e) => { colorSpace = e.detail.space; embedMethod = colorSpace; embedSelection = 'avg'; loadGallery(); }}
              on:setX={(e) => { selectedAxisX = e.detail.id }}
              on:setY={(e) => { selectedAxisY = e.detail.id }}
              on:create={(e) => { const ax = e.detail; if (ax && ax.id) { axes = [ax, ...axes] } }}
              on:delete={(e) => { axes = axes.filter(a => a.id !== e.detail.id) }}
              on:rename={(e) => { axes = axes.map(a => a.id === e.detail.id ? { ...a, name: e.detail.name } : a) }}
            />
            <Callout title="Create personalized axes">
              <ul class="list-disc list-inside">
                <li> Start with an initial projection, </li>
                <li> Define new axes by selecting extremal examples in the scatterplot using the lasso tool </li>
                <li> Drag new axes to reorganize scatterplot </li>
              </ul>
            </Callout>
          </div>
          <div class="flex-1 min-w-0 ml-2">
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
            on:saveSelection={(e) => { const sel = e.detail; if (sel && sel.id) { selections = [sel, ...selections] } }}
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
        </section>
      {/if}

      

      <!-- Rightmost: saved combined selections -->
      {#if false}
      {/if}
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
          <aside class="shrink-0" style={`width:${rightWidth}px;min-width:${minRight}px`}>
          <div class="tile tile-primary">
            <div class="tile-content" style={`height:${rightTileH}px`}>
              <div class="panel-actions"><button class="btn btn-xs btn-ui-secondary" title="Minimize" on:click={() => { rightCollapsed = true }}>–</button></div>
              <div class="tile-header mb-2 flex items-center gap-2"><span class="i-heroicons-rectangle-stack text-slate-600" /> Saved combined selections</div>
                <div class="grid gap-2">
                  {#each combinedSelections as cs (cs.id)}
                    <CombinedSelection
                      combined={cs}
                      idToUrl={new Map(allImages.map(i => [i.id, i.url]))}
                      on:apply={(e)=>{
                        const { posIds=[], negIds=[] } = e.detail || {}
                        const next = {}
                        for (const id of allImages.map(i => i.id)) next[id] = undefined
                        for (const id of posIds) next[id] = 'good'
                        for (const id of negIds) next[id] = 'bad'
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
    </div>
  </main>
</div>

<style>
  :global(html, body, #app) { height: 100%; }
</style>
