<script>
  // import ImageGrid from './components/ImageGrid.svelte' // replaced by TextSimilarity view
  import { generateDemoImages, sortBySimilarity, clusterKMeans } from './lib/data'
  import { onMount, tick } from 'svelte'
  import ScatterMinimap from './components/ScatterMinimap.svelte'
  import AnimatedMinimap from './components/AnimatedMinimap.svelte'
  import HoverGridMinimap from './components/HoverGridMinimap.svelte'
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
  const API_BASE = (import.meta.env && import.meta.env.VITE_API_BASE) ? import.meta.env.VITE_API_BASE : 'http://localhost:5001'
  let datasetPath = '' // leave blank to let server pick sample
  // Radio selection and effective embed method
  let embedSelection = 'avg' // 'avg' | 'clip' | 'dino' | 'dift_sd'
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
  let showTextOverlay = false
  let textSimilarities = {}
  // Split pane state
  let leftWidth = 760
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
    if (m.startsWith('dift_sd_part')) return `Local composition ${m.replace('dift_sd_part','part ')}`
    return m
  }

  // Concepts: saved sets of labels tied to an embedding method
  // Shape: { id, name, method, good: string[], bad: string[] }
  let concepts = []
  // Saved combined concepts for the right panel
  let combinedConcepts = []
  let composerLoadChain = null

  function currentMethodLabel() {
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

  function prefixUrl(u) {
    if (!u) return ''
    if (u.startsWith('http://') || u.startsWith('https://')) return u
    if (u.startsWith('/')) return API_BASE + u
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
    } catch (e) { /* ignore */ }
    loadGallery()
  })

  $: (function persistConcepts(c) {
    try { localStorage.setItem('promptherder.concepts', JSON.stringify(c)) } catch (_) {}
  })(concepts)
  $: (function persistCombined(c) {
    try { localStorage.setItem('promptherder.combined', JSON.stringify(c)) } catch (_) {}
  })(combinedConcepts)

  function onEmbedChange() {
    // Compute effective method; if DIFT selected without part, don't fetch yet
    if (embedSelection === 'dift_sd' && !diftPart) {
      warningMsg = 'Select a DIFT part (3x3) to load embeddings'
      embedMethod = ''
      return
    }
    warningMsg = ''
    embedMethod = embedSelection === 'dift_sd' ? `dift_sd_part${diftPart}` : embedSelection
    loadGallery()
  }

  function onScribbleLabel(e) {
    const { ids, label } = e.detail
    // Update external label object
    for (const id of ids) {
      labelDB[id] = label
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
    fetch(`${API_BASE}/text_force`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, rect, embed: embedSelection === 'dift_sd' ? 'clip' : embedSelection, method: 'pca', alpha: 0.25 })
    }).then(async (res) => {
      if (!res.ok) throw new Error('text_force failed')
      const data = await res.json()
      const sims = data.similarities || []
      const coords = data.coords || []
      const packed = data.packed || []
      prevImages = allImages
      allImages = allImages.map((it, idx) => ({
        ...it,
        x: coords[idx] ? coords[idx][0] : it.x,
        y: coords[idx] ? coords[idx][1] : it.y,
        gx: packed[idx] ? packed[idx][0] : it.gx,
        gy: packed[idx] ? packed[idx][1] : it.gy,
      }))
      textSimilarities[text] = sims
    }).catch((err) => {
      console.error('text_force error', err)
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
      <div class="flex items-center gap-2 mb-2 text-sm">
        <button class="px-2 py-1 border rounded inline-flex items-center gap-1" on:click={() => (scribbleEnabled = !scribbleEnabled)}>
          <span class="i-heroicons-pencil-square" /> {scribbleEnabled ? 'Disable Labeling' : 'Enable Labeling'}
        </button>
        <!-- <button class="px-2 py-1 border rounded ml-2 inline-flex items-center gap-1" on:click={async () => { showTextOverlay = !showTextOverlay; if (showTextOverlay) { await tick(); if (textOverlayRef && textOverlayRef.startPlacing) textOverlayRef.startPlacing() } }}>
          <span class="i-heroicons-rectangle-group" /> {showTextOverlay ? 'Hide Text Tool' : 'Add Text'}
        </button> -->
        {#if scribbleEnabled}
          <button class="px-2 py-1 rounded text-white inline-flex items-center gap-1" style="background:#16a34a" on:click={() => (scribbleLabel = 'good')} aria-label="Good">
            <span class="i-heroicons-hand-thumb-up" /> Positive
          </button>
          <button class="px-2 py-1 rounded text-white inline-flex items-center gap-1" style="background:#dc2626" on:click={() => (scribbleLabel = 'bad')} aria-label="Bad">
            <span class="i-heroicons-hand-thumb-down" /> Negative
          </button>
          <!-- <label class="ml-2 inline-flex items-center gap-2">
            <span class="i-heroicons-arrows-pointing-out" /> Radius
            <input type="range" min="4" max="60" step="1" bind:value={scribbleRadius} class="align-middle" />
          </label> -->
          <button class="px-2 py-1 border rounded inline-flex items-center gap-1" on:click={clearAllLabels}>
            <span class="i-heroicons-trash" /> Clear
          </button>
          <button class="px-2 py-1 border rounded inline-flex items-center gap-1" on:click={createConcept}>
            <span class="i-heroicons-light-bulb" /> Create concept
          </button>
        {/if}
      </div>
      {#if embedSelection === 'dift_sd'}
        <div class="text-sm mb-1">Image-part selection</div>
        <DiftPartSelector on:select={(e) => { diftPart = e.detail.part; embedMethod = `dift_sd_part${diftPart}`; embedSelection = 'dift_sd'; loadGallery(); }} selected={diftPart} />
      {/if}
      <div class="relative" style="width:700px;height:700px;">
        <!-- Method selection inside minimap -->
        <div class="absolute top-1 left-1 z-10 bg-white/90 rounded shadow px-2 py-1 text-xs flex items-center gap-2">
          {#each ['avg','clip','dino','dift_sd'] as m}
            <label class="inline-flex items-center gap-1 cursor-pointer">
              <input type="radio" name="embed" value={m} bind:group={embedSelection} on:change={onEmbedChange} />
              <span class="inline-flex items-center gap-1">
                {#if m==='avg'}<span class="i-heroicons-adjustments-horizontal" />{/if}
                {#if m==='clip'}<span class="i-heroicons-command-line" />{/if}
                {#if m==='dino'}<span class="i-heroicons-cube-transparent" />{/if}
                {#if m==='dift_sd'}<span class="i-heroicons-rectangle-stack" />{/if}
                {methodAlias(m)}
              </span>
            </label>
          {/each}
        </div>
        {#if scribbleEnabled}
          <AnimatedMinimap
            items={allImages.map(i => ({ id: i.id, url: i.url, gx: i.gx, gy: i.gy, x: i.x, y: i.y }))}
            prevItems={prevImages.map(i => ({ id: i.id, url: i.url, gx: i.gx, gy: i.gy, x: i.x, y: i.y }))}
            width={700}
            height={700}
            gridSize={gridSize}
            bind:this={minimapRef}
            enableScribble={scribbleEnabled}
            activeLabel={scribbleLabel}
            brushRadiusPx={scribbleRadius}
            bind:labels={labelsMap}
            on:label={onScribbleLabel}
          />
        {:else}
          <HoverGridMinimap
            items={allImages.map(i => ({ id: i.id, url: i.url, gx: i.gx, gy: i.gy, x: i.x, y: i.y }))}
            width={700}
            height={700}
            gridSize={gridSize}
            viewFrac={0.35}
            experimentalLocalPacking={true}
          />
        {/if}
        <!-- {#if showTextOverlay}
          <MinimapTextOverlay
            bind:this={textOverlayRef}
            width={700}
            height={700}
            on:confirmRegion={onConfirmRegion}
          />
        {/if} -->
      </div>
      <div class="mt-2 text-xs text-gray-600">Positive: {goodCount} • Negative: {badCount}</div>

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
