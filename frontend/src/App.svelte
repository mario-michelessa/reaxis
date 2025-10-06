<script>
  import ImageGrid from './components/ImageGrid.svelte'
  import { generateDemoImages, sortBySimilarity, clusterKMeans } from './lib/data'
  import { onMount, tick } from 'svelte'
  import ScatterMinimap from './components/ScatterMinimap.svelte'
  import AnimatedMinimap from './components/AnimatedMinimap.svelte'
  import DiftPartSelector from './components/DiftPartSelector.svelte'
  import MinimapTextOverlay from './components/MinimapTextOverlay.svelte'

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

  // DIFT part selector (for UI display only)
  let diftPart = '' // e.g., '11', '12', ..., '33'

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
    loadGallery()
  })

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
    // Demo: adjust positions of points inside rect toward its center, to trigger animation
    const cx = rect.x + rect.w / 2
    const cy = rect.y + rect.h / 2
    const moved = new Set()
    const newItems = allImages.map((it) => {
      const inside = it.gx >= rect.x && it.gx <= rect.x + rect.w && it.gy >= rect.y && it.gy <= rect.y + rect.h
      if (!inside) return it
      moved.add(it.id)
      const ax = it.gx + (cx - it.gx) * 0.25
      const ay = it.gy + (cy - it.gy) * 0.25
      return { ...it, gx: ax, gy: ay, x: ax, y: ay }
    })
    // Trigger animation: prev -> new
    prevImages = allImages
    allImages = newItems
    console.log('[text-minimap] confirmRegion', rect, text, 'moved', moved.size)
  }
</script>

<div class="min-h-screen bg-gray-50 text-gray-900">
  <header class="sticky top-0 z-10 bg-white shadow-sm">
    <div class="container mx-auto px-4 py-3 flex items-center gap-4">
      <div class="i-heroicons-photo inline-block text-blue-600 text-2xl" />
      <h1 class="text-xl font-semibold">Image Gallery</h1>
      <div class="flex-1" />
      <div class="flex items-center gap-2">
        <input class="px-2 py-1 border rounded bg-white text-gray-900 border-gray-300 w-60" type="text" placeholder="Dataset path (optional)" bind:value={datasetPath} />
        <fieldset class="flex items-center gap-3">
          <legend class="sr-only">Embedding</legend>
          {#each ['avg','clip','dino','dift_sd'] as m}
            <label class="inline-flex items-center gap-1">
              <input type="radio" name="embed" value={m} bind:group={embedSelection} on:change={onEmbedChange} />
              <span class="text-sm">{m}</span>
            </label>
          {/each}
        </fieldset>
        <button class="btn" on:click={loadGallery}>Reload</button>
        <select class="px-2 py-1 border rounded bg-white text-gray-900 border-gray-300" bind:value={classFilter} on:change={applyFilters}>
          {#each classes as c}
            <option value={c}>{c}</option>
          {/each}
        </select>
        <select class="px-2 py-1 border rounded bg-white text-gray-900 border-gray-300" bind:value={viewMode} on:change={applyFilters}>
          <option value="gallery">Gallery</option>
          <option value="cluster">Cluster</option>
          <option value="similarity">Similarity</option>
        </select>
        <button class="btn" on:click={selectAllCurrent}>Select All</button>
        <button class="btn" on:click={clearSelection}>Clear</button>
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

  <main class="container mx-auto px-4 py-6 flex gap-4">
    <!-- Left minimap -->
    <aside class="shrink-0">
      <div class="text-sm mb-2">Minimap</div>
      <div class="flex items-center gap-2 mb-2 text-sm">
        <button class="px-2 py-1 border rounded" on:click={() => (scribbleEnabled = !scribbleEnabled)}>
          {scribbleEnabled ? 'Disable Scribble' : 'Enable Scribble'}
        </button>
        <span class="ml-1">Brush:</span>
        <button class="px-2 py-1 rounded text-white" style="background:#16a34a" on:click={() => (scribbleLabel = 'good')} aria-label="Good">Good</button>
        <button class="px-2 py-1 rounded text-white" style="background:#dc2626" on:click={() => (scribbleLabel = 'bad')} aria-label="Bad">Bad</button>
        <label class="ml-2">Radius
          <input type="range" min="4" max="60" step="1" bind:value={scribbleRadius} class="align-middle ml-1" />
        </label>
        <button class="px-2 py-1 border rounded" on:click={clearAllLabels}>Clear</button>
        <button class="px-2 py-1 border rounded ml-2" on:click={async () => { showTextOverlay = !showTextOverlay; if (showTextOverlay) { await tick(); if (textOverlayRef && textOverlayRef.startPlacing) textOverlayRef.startPlacing() } }}>
          {showTextOverlay ? 'Hide Text Tool' : 'Add Text Region'}
        </button>
      </div>
      {#if embedSelection === 'dift_sd'}
        <div class="text-sm mb-1">DIFT Part</div>
        <DiftPartSelector on:select={(e) => { diftPart = e.detail.part; embedMethod = `dift_sd_part${diftPart}`; embedSelection = 'dift_sd'; loadGallery(); }} selected={diftPart} />
      {/if}
      <div class="relative" style="width:700px;height:700px;">
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
        {#if showTextOverlay}
          <MinimapTextOverlay
            bind:this={textOverlayRef}
            width={700}
            height={700}
            on:confirmRegion={onConfirmRegion}
          />
        {/if}
      </div>
      <div class="mt-2 text-xs text-gray-600">Good: {goodCount} • Bad: {badCount}</div>

      
    </aside>

    <!-- Main gallery area -->
    <section class="flex-1">
    {#if viewMode === 'cluster'}
      {#each clusters as cluster}
        <section class="mb-8">
          <h2 class="text-lg font-semibold mb-3">{cluster.label} <span class="chip ml-2">{cluster.items.length}</span></h2>
          <ImageGrid items={cluster.items} {selected} onToggleSelect={toggleSelect} onClick={(item) => (viewMode === 'similarity' ? setReference(item) : toggleSelect(item.id))} />
        </section>
      {/each}
    {:else}
      <ImageGrid items={filtered} {selected} onToggleSelect={toggleSelect} onClick={(item) => (viewMode === 'similarity' ? setReference(item) : toggleSelect(item.id))} />
    {/if}

    {#if viewMode === 'similarity'}
      <div class="mt-4 flex items-center gap-2">
        <span class="text-sm">Reference:</span>
        {#if refId}
          <span class="chip">{filtered.find(i => i.id === refId)?.label || refId}</span>
          <button class="btn" on:click={() => (refId = null, applyFilters())}>Clear Reference</button>
        {:else}
          <span class="text-sm text-gray-500">Click an image to set reference</span>
        {/if}
      </div>
    {/if}

    <div class="mt-6 text-sm text-gray-600">Selected: {selected.size}</div>
    </section>
  </main>
</div>

<style>
  :global(html, body, #app) { height: 100%; }
</style>
