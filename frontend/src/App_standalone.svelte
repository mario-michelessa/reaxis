<script>
  import { generateDemoImages } from './lib/data'
  import { onMount } from 'svelte'
  import AxesMinimap from './components/AxesMinimap.svelte'
  import AxesPanel from './components/AxesPanel.svelte'
  import Callout from './components/Callout.svelte'
  import CombinedSelection from './components/CombinedSelection.svelte'
  import SelectionComposer from './components/SelectionComposer.svelte'

  // Standalone mode: no backend. Supports multiple datasets under ./datasets/<name>/ with a gallery.json. Falls back to ./gallery.json or demo data.
  let allImages = []
  let prevImages = []
  const galleryCache = new Map()
  const metadataCache = new Map()
  let selected = new Set()

  let warningMsg = ''
  let labelDB = {}
  $: posCount = Object.values(labelDB).filter(v => v === 'pos').length
  $: negCount = Object.values(labelDB).filter(v => v === 'neg').length

  // Static datasets discovery and selection
  let datasets = [] // [{label, value}]
  let datasetName = '' // value in datasets
  function resetAllStateForDatasetChange() {
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
    galleryCache.clear()
  }

  let minimapContainerRef
  let leftWidth = 1200
  let leftTileH = 920
  let leftCollapsed = false
  let middleWidth = 700
  let middleTileH = 600
  let middleCollapsed = true
  let rightWidth = 320
  let rightTileH = 600
  let rightCollapsed = true
  let minLeft = 500
  let maxLeft = 2000
  let minRight = 180

  let tileDrag = null
  function startLeftTileResize(e) { tileDrag = { which: 'left', startX: e.clientX, startY: e.clientY, startW: leftWidth, startH: leftTileH } }
  function startMiddleTileResize(e) { tileDrag = { which: 'middle', startX: e.clientX, startY: e.clientY, startW: middleWidth, startH: middleTileH } }
  function startRightTileResize(e) { tileDrag = { which: 'right', startX: e.clientX, startY: e.clientY, startW: rightWidth, startH: rightTileH } }
  function onDrag(e) {
    if (!tileDrag) return
    const dx = e.clientX - tileDrag.startX
    const dy = e.clientY - tileDrag.startY
    if (tileDrag.which === 'left') { leftWidth = Math.max(minLeft, Math.min(maxLeft, tileDrag.startW + dx)); leftTileH = Math.max(300, tileDrag.startH + dy) }
    if (tileDrag.which === 'middle') { middleWidth = Math.max(400, tileDrag.startW + dx); middleTileH = Math.max(300, tileDrag.startH + dy) }
    if (tileDrag.which === 'right') { rightWidth = Math.max(minRight, tileDrag.startW + dx); rightTileH = Math.max(300, tileDrag.startH + dy) }
  }
  function endDrag() { tileDrag = null }

  let windowWidth = 0
  $: minimapW = (() => { void leftWidth; void windowWidth; void leftTileH; const w = minimapContainerRef ? Math.floor(minimapContainerRef.clientWidth - 470) : 700; return Math.max(300, Math.min(2000, w)) })()
  $: minimapH = (() => { void leftTileH; const h = minimapContainerRef ? Math.floor(minimapContainerRef.clientHeight - 50) : 700; return Math.max(300, Math.min(2000, h)) })()

  // Projections and axes
  let axes = []
  let selectedAxisX = null
  let selectedAxisY = null
  let customProjections = []
  let embedSelection = 'color_rgb'
  let embedMethod = 'color_rgb'
  let selections = []
  let combinedSelections = []
  const projectionOptions = [
    { value: 'color_rgb', label: 'Color' },
    { value: 'clip', label: 'Semantic' },
    { value: 'shape', label: 'Shape' },
    { value: 'meta', label: 'Metadata' },
  ]

  function axisPrefixForMethod(method) {
    if (!method) return 'Axis'
    if (method === 'color_rgb') return 'RGB'
    if (method === 'color_hsv') return 'HSV'
    if (method === 'color_lch') return 'LCh'
    if (method === 'clip') return 'CLIP'
    if (method === 'dino') return 'DINO'
    if (String(method).startsWith('dift_sd_part')) return `DIFT ${String(method).replace('dift_sd_part','')}`
    if (method === 'dift_sd') return 'DIFT'
    return method
  }

  function idMatchesProjection(groupKey, axisId) {
    if (!axisId || !groupKey) return false
    if (groupKey === 'color_rgb') return axisId.startsWith('axis:color_lch:') || axisId.startsWith('axis:color_hsv:') || axisId.startsWith('axis:color_rgb:')
    if (groupKey === 'shape') return axisId.startsWith('axis:dino:') || axisId.startsWith('axis:dift_sd_part') || axisId.startsWith('axis:dift_sd:')
    if (groupKey === 'clip') return axisId.startsWith('axis:clip:')
    if (groupKey === 'meta') return axisId.startsWith('axis:meta:')
    return false
  }

  function ensureDefaultAxesForCurrentProjection() {
    const methodName = embedMethod
    if (!methodName) return
    const prefix = axisPrefixForMethod(methodName)
    const idX = `axis:${methodName}:x`
    const idY = `axis:${methodName}:y`
    const coordsX = {}
    const coordsY = {}
    for (const it of allImages) { coordsX[it.id] = Number(it.x ?? 0); coordsY[it.id] = Number(it.y ?? 0) }
    const projGroup = (embedSelection && String(embedSelection).startsWith('custom:')) ? embedSelection : (embedSelection || 'color_rgb')
    let next = axes
    const axX = { id: idX, name: `${prefix} X`, coords: coordsX, group: projGroup }
    const axY = { id: idY, name: `${prefix} Y`, coords: coordsY, group: projGroup }
    const hasX = next.some(a => a.id === idX)
    const hasY = next.some(a => a.id === idY)
    if (hasX) next = next.map(a => a.id === idX ? axX : a); else next = [...next, axX]
    if (hasY) next = next.map(a => a.id === idY ? axY : a); else next = [...next, axY]
    axes = next
    // Always select the axes for the current method to ensure position updates
    selectedAxisX = idX
    selectedAxisY = idY
  }

  // Thumbnails: passthrough in standalone (no /thumb endpoint)
  function toThumbUrl(u) { return u || '' }
  function datasetPrefix() { return datasetName ? `./datasets/${datasetName}/` : './' }
  function prefixUrl(u) {
    if (!u) return ''
    if (u.startsWith('http://') || u.startsWith('https://')) return u
    if (u.startsWith('/')) return u // absolute path served by host
    // relative path inside dataset folder
    return datasetPrefix() + u.replace(/^\.\//, '')
  }

  // Prefetch basic images (no-op safety)
  const _prefetched = new Set()
  function schedulePrefetch(urls) {
    if (!Array.isArray(urls) || urls.length === 0) return
    const run = () => {
      let count = 0
      for (const u of urls) {
        if (!u || _prefetched.has(u)) continue
        try { const img = new Image(); img.decoding = 'async'; img.loading = 'eager'; img.referrerPolicy = 'no-referrer'; img.src = u; _prefetched.add(u); count++; if (count >= 12) break } catch (_) {}
      }
      const remaining = urls.filter(u => u && !_prefetched.has(u))
      if (remaining.length > 0) setTimeout(run, 80)
    }
    if ('requestIdleCallback' in window) { try { window.requestIdleCallback(run, { timeout: 300 }) } catch (_) { setTimeout(run, 50) } } else { setTimeout(run, 50) }
  }

  let _loadingGuard = 0
  async function loadStaticGallery() {
    const ds = datasetName || '_default'
    const targetMethod = String(embedMethod || '')
    const cacheKey = `standalone|${ds}|${targetMethod}|pca`
    const myGuard = ++_loadingGuard
    if (galleryCache.has(cacheKey)) {
      const cached = galleryCache.get(cacheKey)
      warningMsg = ''
      prevImages = allImages
      allImages = cached.items
      try { schedulePrefetch(allImages.map(it => it.url)) } catch (_) {}
      ensureDefaultAxesForCurrentProjection()
      return
    }
    try {
      // Prefer method-specific gallery if available, otherwise fall back to default gallery.json
      const base = datasetName ? `./datasets/${datasetName}/` : './'
      const preferred = `${base}gallery_${targetMethod}.json`
      const fallback = `${base}gallery.json`
      let res = await fetch(preferred, { cache: 'no-cache' }).catch(() => null)
      // For part-specific shape methods, do not silently fall back; surface only if preferred is missing
      const isShapePart = targetMethod.startsWith('dift_sd_part') || targetMethod === 'dino'
      if (!(res && res.ok) && !isShapePart) {
        res = await fetch(fallback, { cache: 'no-cache' }).catch(() => null)
      }
      if (res && res.ok) {
        const data = await res.json()
        const items = Array.isArray(data.items) ? data.items : []
        // Guard against out-of-order responses: if method switched while loading, ignore this response
        if (myGuard !== _loadingGuard || String(embedMethod || '') !== targetMethod) {
          return
        }
        prevImages = allImages
        allImages = items.map((it) => ({
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
        // Keep the current target method to avoid accidentally reverting due to fallback metadata
        embedMethod = targetMethod
        // Ingest metadata axes if present
        try {
          const metaAxes = Array.isArray(data.metadata_axes) ? data.metadata_axes : []
          if (metaAxes.length > 0) {
            const incoming = metaAxes.map(a => ({ id: a.id, name: a.name || a.id, coords: a.coords || {}, labels: a.labels || [], label_positions: a.label_positions || [], group: 'meta' }))
            const existing = new Map(axes.map(a => [a.id, a]))
            for (const ax of incoming) { if (!existing.has(ax.id)) existing.set(ax.id, ax) }
            axes = Array.from(existing.values())
          }
        } catch (_) {}
        // Merge in dataset-level metadata axes from gallery_metadata.json (cached by dataset)
        try {
          if (metadataCache.has(ds)) {
            const incoming = metadataCache.get(ds) || []
            const existing = new Map(axes.map(a => [a.id, a]))
            for (const ax of incoming) { if (ax && ax.id && !existing.has(ax.id)) existing.set(ax.id, ax) }
            axes = Array.from(existing.values())
          } else {
            const base = datasetName ? `./datasets/${datasetName}/` : './'
            const mres = await fetch(`${base}gallery_metadata.json`, { cache: 'no-cache' }).catch(() => null)
            if (mres && mres.ok) {
              const mjson = await mres.json()
              const metaAxes2 = Array.isArray(mjson.metadata_axes) ? mjson.metadata_axes : []
              const incoming = metaAxes2.map(a => ({ id: a.id, name: a.name || a.id, coords: a.coords || {}, labels: a.labels || [], label_positions: a.label_positions || [], group: 'meta' }))
              metadataCache.set(ds, incoming)
              const existing = new Map(axes.map(a => [a.id, a]))
              for (const ax of incoming) { if (ax && ax.id && !existing.has(ax.id)) existing.set(ax.id, ax) }
              axes = Array.from(existing.values())
            } else {
              metadataCache.set(ds, [])
            }
          }
        } catch (_) {}
        galleryCache.set(cacheKey, { items: allImages })
        try { schedulePrefetch(allImages.map(it => it.url)) } catch (_) {}
        ensureDefaultAxesForCurrentProjection()
        return
      }
    } catch (_) { /* fall through */ }
    // Fallback to demo data
    warningMsg = ''
    prevImages = []
    allImages = generateDemoImages(48).map((it) => ({ ...it, x: it.embed[0], y: it.embed[1], gx: it.embed[0], gy: it.embed[1] }))
    ensureDefaultAxesForCurrentProjection()
  }
  onMount(async () => {
    // Load dataset manifest if present
    try {
      let manifest = null
      const tryUrls = ['./datasets/index.json', './datasets.json']
      for (const u of tryUrls) {
        try { const r = await fetch(u, { cache: 'no-cache' }); if (r.ok) { manifest = await r.json(); break } } catch (_) {}
      }
      if (manifest) {
        const list = Array.isArray(manifest) ? manifest : []
        const mapped = list.map(d => {
          if (typeof d === 'string') return { label: d, value: d }
          if (d && typeof d === 'object') {
            const label = d.label || d.name || d.id || d.path || 'Dataset'
            const value = d.value || d.id || d.path || d.name || label
            return { label, value }
          }
          return null
        }).filter(Boolean)
        if (mapped.length > 0) {
          datasets = mapped
          if (!datasetName) datasetName = datasets[0].value
        }
      }
    } catch (_) {}
    await loadStaticGallery()
  })

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
      <div class="brand-name">ReQuest (Standalone)</div>
      <!-- Dataset selector (static manifest) -->
      <div class="ml-4 inline-flex items-center gap-2">
        <label for="dataset-select" class="text-sm text-gray-700">Dataset</label>
        <select id="dataset-select" class="text-sm" on:change={async (e)=>{ const val = e.currentTarget.value; if ((datasetName||'')===(val||'')) return; datasetName = val; resetAllStateForDatasetChange(); await loadStaticGallery() }}>
          {#each datasets as d}
            <option value={d.value} selected={(datasetName||'')===(d.value||'')}>{d.label}</option>
          {/each}
        </select>
      </div>
      <div class="flex-1" />
    </div>
  </header>

  <main class="app-main w-full py-6">
    <div class="panels-row flex flex-nowrap overflow-x-auto" >
      {#if leftCollapsed}
        <div class="shrink-0" style="width:50px;">
          <button class="collapsed-handle rotate tile-header" title="Define projection" on:click={() => { leftCollapsed = false; leftWidth = Math.max(minLeft, leftWidth) }}>Define projection</button>
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
                        const cp = (customProjections || []).find(p => p.id === sel)
                        if (cp) {
                          // Custom projection: assign axes only, no reload
                          embedSelection = cp.id
                          if (cp.xAxisId) selectedAxisX = cp.xAxisId
                          if (cp.yAxisId) selectedAxisY = cp.yAxisId
                          return
                        }
                        // Metadata: don't change method, just pick two meta axes if available
                        if (sel === 'meta') {
                          embedSelection = 'meta'
                          const underGroup = axes.filter(a => (a?.group === 'meta') || idMatchesProjection('meta', a?.id || ''))
                          if (underGroup.length >= 2) { selectedAxisX = underGroup[0].id; selectedAxisY = underGroup[1].id }
                          return
                        }
                        // DIFT/DINO part selector emits either 'dino' or 'dift_sd_partXY'
                        if (sel && (sel === 'dino' || String(sel).startsWith('dift_sd_part'))) {
                          embedSelection = 'shape'
                          if (embedMethod !== sel) { embedMethod = sel; loadStaticGallery() }
                          return
                        }
                        // If user selects the Shape radio (no specific part yet), default to 'dino'
                        if (sel === 'shape') {
                          embedSelection = 'shape'
                          if (embedMethod !== 'dino') { embedMethod = 'dino'; loadStaticGallery() }
                          return
                        }
                        // Built-ins: color_rgb or clip
                        if (sel === 'color_rgb' || sel === 'clip') {
                          embedSelection = sel
                          if (embedMethod !== sel) { embedMethod = sel; loadStaticGallery() }
                          return
                        }
                        // Fallback: treat selection as method key
                        embedSelection = sel
                        if (sel && embedMethod !== sel) { embedMethod = sel; loadStaticGallery() }
                      }}
                      on:setX={(e) => { selectedAxisX = e.detail.id }}
                      on:setY={(e) => { selectedAxisY = e.detail.id }}
                      on:create={(e) => { const ax = e.detail; if (ax && ax.id) { const groupKey = (embedSelection && String(embedSelection).startsWith('custom:')) ? embedSelection : (embedSelection || 'color_rgb'); const withGroup = { ...ax, group: ax.group || groupKey }; axes = [withGroup, ...axes] } }}
                      on:delete={(e) => { axes = axes.filter(a => a.id !== e.detail.id) }}
                      on:rename={(e) => { axes = axes.map(a => a.id === e.detail.id ? { ...a, name: e.detail.name } : a) }}
                      on:addCustomProjection={(e) => { const name = (e.detail?.name || '').trim(); if (!name) return; const id = `custom:${Date.now()}`; const xAxisId = selectedAxisX; const yAxisId = selectedAxisY; customProjections = [{ id, name, xAxisId, yAxisId }, ...customProjections]; embedSelection = id; if (xAxisId) selectedAxisX = xAxisId; if (yAxisId) selectedAxisY = yAxisId }}
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
                      on:axesChange={(e)=>{ selectedAxisX = e.detail.selectedX; selectedAxisY = e.detail.selectedY }}
                      on:label={onScribbleLabel}
                      on:create={(e) => { const ax = e.detail; if (ax && ax.id) { const groupKey = (embedSelection && String(embedSelection).startsWith('custom:')) ? embedSelection : (embedSelection || 'color_rgb'); const withGroup = { ...ax, group: ax.group || groupKey }; axes = [withGroup, ...axes] } }}
                      on:saveSelection={(e) => { const sel = e.detail; if (sel && sel.id) { selections = [sel, ...selections] } }}
                    />
                  </div>
                </div>
              </div>
              <div class="tile-resize-handle" title="Resize" on:mousedown={startLeftTileResize}></div>
            </div>
          </div>
        </aside>
      {/if}

      {#if middleCollapsed}
        <div class="shrink-0 pr-2" style="width:50px;">
          <button class="collapsed-handle rotate tile-header" title="Define Selections" on:click={() => { middleCollapsed = false }}>
            Define selections
          </button>
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

      {#if true}
        {#if rightCollapsed}
          <div class="shrink-0" style="width:50px;">
            <button class="collapsed-handle rotate tile-header" title="Show Combined selections" on:click={() => { rightCollapsed = false }}>
              Combined selections
            </button>
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
    </div>
  </main>
</div>

<style>
  :global(html, body, #app) { height: 100%; }
</style>
