<script>
  import { generateDemoImages } from './lib/data'
  import { onMount } from 'svelte'
  import AxesMinimap from './components/AxesMinimap.svelte'
  import PromptSidebar from './components/PromptSidebar.svelte'

  // Standalone mode: no backend. Supports datasets under ./datasets/<name>/.
  let allImages = []
  const galleryCache = new Map()
  const metadataCache = new Map()

  let warningMsg = ''
  let labelDB = {}

  // Dataset discovery and selection
  let datasets = []
  let datasetName = ''

  // Prompt-driven axes and minimap state
  let axes = []
  let selectedAxisX = null
  let selectedAxisY = null
  let embedMethod = 'siglip2'

  // Layout sizing
  let minimapContainerRef
  let windowWidth = 0
  let windowHeight = 0
  let leftPanelWidth = 370

  $: panelHeight = Math.max(620, windowHeight - 170)
  $: minimapW = (() => {
    void windowWidth
    void panelHeight
    const w = minimapContainerRef ? Math.floor(minimapContainerRef.clientWidth - 75) : 980
    return Math.max(420, Math.min(2400, w))
  })()
  $: minimapH = (() => {
    void panelHeight
    const h = minimapContainerRef ? Math.floor(minimapContainerRef.clientHeight - 74) : 820
    return Math.max(420, Math.min(2000, h))
  })()

  function resetAllStateForDatasetChange() {
    allImages = []
    axes = []
    selectedAxisX = null
    selectedAxisY = null
    embedMethod = 'siglip2'
    labelDB = {}
    warningMsg = ''
    galleryCache.clear()
  }

  function ensureDefaultAxesForCurrentProjection() {
    const methodName = String(embedMethod || 'siglip2')
    const idX = `axis:${methodName}:x`
    const idY = `axis:${methodName}:y`

    const coordsX = {}
    const coordsY = {}
    for (const item of allImages) {
      coordsX[item.id] = Number(item.x ?? 0)
      coordsY[item.id] = Number(item.y ?? 0)
    }

    const defaults = new Map(axes.map((axis) => [axis.id, axis]))
    defaults.set(idX, { id: idX, name: 'Embedding X', coords: coordsX, group: 'base' })
    defaults.set(idY, { id: idY, name: 'Embedding Y', coords: coordsY, group: 'base' })
    axes = Array.from(defaults.values())

    if (!selectedAxisX || !defaults.has(selectedAxisX)) selectedAxisX = idX
    if (!selectedAxisY || !defaults.has(selectedAxisY)) selectedAxisY = idY
  }

  function datasetPrefix() {
    return datasetName ? `./datasets/${datasetName}/` : './'
  }

  function prefixUrl(url) {
    if (!url) return ''
    if (url.startsWith('http://') || url.startsWith('https://')) return url
    if (url.startsWith('/')) return url
    return datasetPrefix() + url.replace(/^\.\//, '')
  }

  const prefetched = new Set()
  function schedulePrefetch(urls) {
    if (!Array.isArray(urls) || urls.length === 0) return
    const run = () => {
      let count = 0
      for (const url of urls) {
        if (!url || prefetched.has(url)) continue
        try {
          const img = new Image()
          img.decoding = 'async'
          img.loading = 'eager'
          img.referrerPolicy = 'no-referrer'
          img.src = url
          prefetched.add(url)
          count += 1
          if (count >= 12) break
        } catch (_) {}
      }
      const remaining = urls.filter((url) => url && !prefetched.has(url))
      if (remaining.length > 0) setTimeout(run, 80)
    }
    if ('requestIdleCallback' in window) {
      try { window.requestIdleCallback(run, { timeout: 300 }) } catch (_) { setTimeout(run, 50) }
    } else {
      setTimeout(run, 50)
    }
  }

  let loadingGuard = 0
  async function loadStaticGallery() {
    const datasetKey = datasetName || '_default'
    const targetMethod = String(embedMethod || 'siglip2')
    const cacheKey = `standalone|${datasetKey}|${targetMethod}|pca`
    const myGuard = ++loadingGuard

    if (galleryCache.has(cacheKey)) {
      const cached = galleryCache.get(cacheKey)
      warningMsg = ''
      allImages = cached.items
      try { schedulePrefetch(allImages.map((item) => item.url)) } catch (_) {}
      ensureDefaultAxesForCurrentProjection()
      return
    }

    try {
      const base = datasetPrefix()
      const preferred = `${base}gallery_${targetMethod}.json`
      const fallback = `${base}gallery.json`

      let res = await fetch(preferred, { cache: 'no-cache' }).catch(() => null)
      if (!(res && res.ok)) {
        res = await fetch(fallback, { cache: 'no-cache' }).catch(() => null)
      }

      if (res && res.ok) {
        const data = await res.json()
        const items = Array.isArray(data.items) ? data.items : []
        const effectiveEmbedMethod = String(data.embed || targetMethod || 'siglip2')

        // Ignore stale responses from earlier dataset/method requests.
        if (myGuard !== loadingGuard || String(embedMethod || '') !== targetMethod) return
        if (effectiveEmbedMethod) embedMethod = effectiveEmbedMethod

        allImages = items.map((item) => ({
          id: item.id,
          url: prefixUrl(item.url || item.path),
          className: item.className || 'Unknown',
          label: item.id,
          gx: Number(item.gx ?? item.x ?? 0),
          gy: Number(item.gy ?? item.y ?? 0),
          x: Number(item.x ?? item.gx ?? 0),
          y: Number(item.y ?? item.gy ?? 0),
          embed: Array.isArray(item.embed)
            ? item.embed.map((v) => Number(v)).filter((v) => Number.isFinite(v))
            : [Number(item.x ?? item.gx ?? 0), Number(item.y ?? item.gy ?? 0)]
        }))

        // Merge metadata axes from gallery payload.
        try {
          const incomingRaw = Array.isArray(data.metadata_axes) ? data.metadata_axes : []
          if (incomingRaw.length > 0) {
            const incoming = incomingRaw.map((axis) => ({
              id: axis.id,
              name: axis.name || axis.id,
              coords: axis.coords || {},
              labels: axis.labels || [],
              label_positions: axis.label_positions || [],
              group: 'meta'
            }))
            const existing = new Map(axes.map((axis) => [axis.id, axis]))
            for (const axis of incoming) {
              if (axis && axis.id && !existing.has(axis.id)) existing.set(axis.id, axis)
            }
            axes = Array.from(existing.values())
          }
        } catch (_) {}

        // Merge dataset-level metadata axes from gallery_metadata.json.
        try {
          if (metadataCache.has(datasetKey)) {
            const incoming = metadataCache.get(datasetKey) || []
            const existing = new Map(axes.map((axis) => [axis.id, axis]))
            for (const axis of incoming) {
              if (axis && axis.id && !existing.has(axis.id)) existing.set(axis.id, axis)
            }
            axes = Array.from(existing.values())
          } else {
            const mres = await fetch(`${base}gallery_metadata.json`, { cache: 'no-cache' }).catch(() => null)
            if (mres && mres.ok) {
              const mjson = await mres.json()
              const incomingRaw = Array.isArray(mjson.metadata_axes) ? mjson.metadata_axes : []
              const incoming = incomingRaw.map((axis) => ({
                id: axis.id,
                name: axis.name || axis.id,
                coords: axis.coords || {},
                labels: axis.labels || [],
                label_positions: axis.label_positions || [],
                group: 'meta'
              }))
              metadataCache.set(datasetKey, incoming)
              const existing = new Map(axes.map((axis) => [axis.id, axis]))
              for (const axis of incoming) {
                if (axis && axis.id && !existing.has(axis.id)) existing.set(axis.id, axis)
              }
              axes = Array.from(existing.values())
            } else {
              metadataCache.set(datasetKey, [])
            }
          }
        } catch (_) {}

        galleryCache.set(cacheKey, { items: allImages })
        try { schedulePrefetch(allImages.map((item) => item.url)) } catch (_) {}
        warningMsg = ''
        ensureDefaultAxesForCurrentProjection()
        return
      }
    } catch (_) {}

    // Fallback to demo data.
    warningMsg = ''
    allImages = generateDemoImages(48).map((item) => ({
      ...item,
      x: item.embed[0],
      y: item.embed[1],
      gx: item.embed[0],
      gy: item.embed[1]
    }))
    ensureDefaultAxesForCurrentProjection()
  }

  onMount(async () => {
    try {
      let manifest = null
      const urls = ['./datasets/index.json', './datasets.json']
      for (const url of urls) {
        try {
          const res = await fetch(url, { cache: 'no-cache' })
          if (res.ok) {
            manifest = await res.json()
            break
          }
        } catch (_) {}
      }

      if (manifest) {
        const list = Array.isArray(manifest) ? manifest : []
        const mapped = list
          .map((d) => {
            if (typeof d === 'string') return { label: d, value: d }
            if (d && typeof d === 'object') {
              const label = d.label || d.name || d.id || d.path || 'Dataset'
              const value = d.value || d.id || d.path || d.name || label
              return { label, value }
            }
            return null
          })
          .filter(Boolean)
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
      for (const update of updates) {
        if (!update) continue
        if (update.label === null || update.label === undefined || update.label === 'none') delete next[update.id]
        else if (update.label === 'pos' || update.label === 'neg') next[update.id] = update.label
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

  function upsertAxisValue(axis) {
    if (!axis || !axis.id) return
    let found = false
    const next = axes.map((existing) => {
      if (existing.id !== axis.id) return existing
      found = true
      return { ...existing, ...axis }
    })
    axes = found ? next : [{ ...axis, group: axis.group || 'prompt' }, ...next]
  }

  function upsertAxis(e) {
    upsertAxisValue(e.detail?.axis)
  }

  function fallbackAxisId(slot, nextAxes) {
    const preferred = `axis:${String(embedMethod || 'siglip2')}:${slot}`
    if (Array.isArray(nextAxes) && nextAxes.some((axis) => axis?.id === preferred)) return preferred
    return Array.isArray(nextAxes) && nextAxes.length > 0 ? nextAxes[0].id : null
  }

  function removeAxisValue(axisId) {
    const id = String(axisId || '').trim()
    if (!id) return
    const next = (axes || []).filter((axis) => axis?.id !== id)
    axes = next
    if (selectedAxisX === id) selectedAxisX = fallbackAxisId('x', next)
    if (selectedAxisY === id) selectedAxisY = fallbackAxisId('y', next)
  }

  function onRemoveAxis(e) {
    removeAxisValue(e.detail?.id)
  }
</script>

<svelte:window bind:innerWidth={windowWidth} bind:innerHeight={windowHeight} />

<div class="app-main text-gray-900">
  <header class="app-header">
    <div class="app-header-inner">
      <div class="i-heroicons-sparkles brand-icon" />
      <div class="brand-name">ReQuest (Standalone)</div>
      <div class="ml-4 inline-flex items-center gap-2">
        <label for="dataset-select" class="text-sm text-gray-700">Dataset</label>
        <select
          id="dataset-select"
          class="text-sm"
          on:change={async (e) => {
            const next = e.currentTarget.value
            if ((datasetName || '') === (next || '')) return
            datasetName = next
            resetAllStateForDatasetChange()
            await loadStaticGallery()
          }}
        >
          {#each datasets as d}
            <option value={d.value} selected={(datasetName || '') === (d.value || '')}>{d.label}</option>
          {/each}
        </select>
      </div>
      <div class="flex-1" />
    </div>
  </header>

  {#if warningMsg}
    <div class="bg-yellow-50 border-l-4 border-yellow-400 text-yellow-800 p-3">
      <div class="text-sm px-4">{warningMsg}</div>
    </div>
  {/if}

  <main class="app-main w-full py-6">
    <div class="workspace-grid px-3">
      <aside class="workspace-sidebar shrink-0" style={`width:${leftPanelWidth}px;min-width:300px;`}>
        <div class="tile tile-primary">
          <div class="tile-header mb-1 flex items-center gap-2">
            <span class="i-heroicons-chat-bubble-left-right text-slate-600" />
            Prompt Axis Builder
          </div>
          <div class="tile-content" style={`height:${panelHeight}px`}>
            <PromptSidebar
              {axes}
              items={allImages}
              selectedX={selectedAxisX}
              selectedY={selectedAxisY}
              on:upsertAxis={upsertAxis}
              on:removeAxis={onRemoveAxis}
              on:setX={(e) => { if (e.detail?.id) selectedAxisX = e.detail.id }}
              on:setY={(e) => { if (e.detail?.id) selectedAxisY = e.detail.id }}
            />
          </div>
        </div>
      </aside>

      <section class="workspace-center min-w-0 flex-1">
        <div class="tile tile-primary">
          <div class="tile-header mb-1 flex items-center gap-2">
            <span class="i-heroicons-chart-bar-square text-slate-600" />
            Scatterplot Workspace
          </div>
          <div class="tile-content flex items-start justify-center" bind:this={minimapContainerRef} style={`height:${panelHeight}px`}>
            <AxesMinimap
              items={allImages.map((item) => ({
                id: item.id,
                url: item.url,
                gx: item.gx,
                gy: item.gy,
                x: item.x,
                y: item.y,
                embed: item.embed
              }))}
              selections={[]}
              axes={axes}
              width={minimapW}
              height={minimapH}
              labels={new Map(Object.entries(labelDB))}
              selectionToolsEnabled={false}
              bind:selectedX={selectedAxisX}
              bind:selectedY={selectedAxisY}
              on:axesChange={(e) => {
                selectedAxisX = e.detail.selectedX
                selectedAxisY = e.detail.selectedY
              }}
              on:label={onScribbleLabel}
              on:create={(e) => {
                const axis = e.detail
                if (!axis || !axis.id) return
                upsertAxisValue({ ...axis, group: axis.group || 'prompt' })
              }}
            />
          </div>
        </div>
      </section>
    </div>
  </main>
</div>

<style>
  :global(html, body, #app) { height: 100%; }

  .workspace-grid {
    display: flex;
    flex-wrap: nowrap;
    gap: 12px;
    align-items: flex-start;
    overflow-x: auto;
  }

  .workspace-sidebar .tile-content {
    overflow: hidden;
  }

  @media (max-width: 1080px) {
    .workspace-grid {
      flex-direction: column;
      overflow-x: visible;
    }

    .workspace-sidebar {
      width: 100% !important;
      min-width: 0 !important;
    }
  }
</style>
