<script>
  import ImageGrid from './ImageGrid.svelte'
  import { buildApiUrl, resolveApiBase } from '../lib/apiBase'

  export let items = [] // full gallery items [{id,url,className,label,gx,gy,x,y}]
  export let apiBase = resolveApiBase()
  export let defaultTopN = 10
  export let onCreateConcept = (goodIds) => {}
  export let onRefreshGallery = async () => {}

  let queryText = ''
  let topN = defaultTopN
  let results = [] // ranked items
  let selected = new Set()
  let busy = false
  let error = ''

  $: resultsTop = results.slice(0, Math.max(1, Number(topN) || 10))

  function toggleSelect(id) {
    if (selected.has(id)) selected.delete(id)
    else selected.add(id)
    selected = new Set(selected)
  }

  async function runSearch() {
    error = ''
    if (!items || items.length === 0) { error = 'No images loaded'; return }
    if (!queryText || !queryText.trim()) { error = 'Enter some text to search'; return }
    busy = true
    try {
      const res = await fetch(buildApiUrl(apiBase, '/text_force'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: queryText.trim(), rect: { x: 0, y: 0, w: 1, h: 1 }, embed: 'siglip2', alpha: 0 })
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      const sims = data?.similarities || []
      const withScore = items.map((it, idx) => ({ ...it, _score: sims[idx] ?? 0 }))
      withScore.sort((a, b) => (b._score - a._score))
      results = withScore
    } catch (e) {
      console.error('text search failed', e)
      error = 'Search failed'
    } finally {
      busy = false
    }
  }

  async function onUpload(e) {
    const file = e.target?.files?.[0]
    if (!file) return
    const form = new FormData()
    form.append('file', file)
    try {
      busy = true
      const res = await fetch(buildApiUrl(apiBase, '/upload'), { method: 'POST', body: form })
      if (!res.ok) throw new Error(await res.text())
      await onRefreshGallery()
    } catch (e) {
      console.error('upload failed', e)
      error = 'Upload failed'
    } finally {
      busy = false
      e.target.value = ''
    }
  }

  function createConceptFromSelection() {
    const ids = Array.from(selected)
    if (ids.length === 0) return
    onCreateConcept(ids)
    selected = new Set()
  }
</script>

<div class="card p-3">
  <div class="flex items-center gap-2 mb-3">
    <div class="i-heroicons-magnifying-glass-20-solid text-gray-600" />
    <input class="flex-1 px-3 py-2 border rounded bg-white" placeholder="Type text to find similar images" bind:value={queryText} on:keydown={(e) => e.key==='Enter' && runSearch()} />
    <button class="btn" on:click={runSearch} disabled={busy}>
      <span class="i-heroicons-sparkles mr-1" /> Search
    </button>
    <label class="btn relative overflow-hidden">
      <span class="i-heroicons-arrow-up-tray mr-1" /> Upload
      <input type="file" accept="image/*" class="absolute inset-0 opacity-0 cursor-pointer" on:change={onUpload} />
    </label>
    <label class="flex items-center gap-2 text-sm ml-2 bg-white">
      Show
      <input class="w-15 px-2 py-1 border rounded bg-white" type="number" min="1" max="100" bind:value={topN} />
      images
    </label>
  </div>
  {#if error}
    <div class="text-sm text-red-600 mb-2">{error}</div>
  {/if}

  <ImageGrid items={resultsTop} {selected} onToggleSelect={toggleSelect} onClick={(item) => toggleSelect(item.id)} />

  <div class="mt-3 flex items-center justify-between text-sm">
    <div class="text-gray-600">Selected: {selected.size}</div>
    <button class="btn" disabled={selected.size===0} on:click={createConceptFromSelection}>
      <span class="i-heroicons-light-bulb mr-1" /> Create Concept
    </button>
  </div>
</div>

<style>
</style>
