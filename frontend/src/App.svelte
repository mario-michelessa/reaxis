<script>
  import { onDestroy, onMount } from 'svelte'
  import AxesMinimap from './components/AxesMinimap.svelte'
  import LoadingGate from './components/LoadingGate.svelte'
  import PromptSidebar from './components/PromptSidebar.svelte'
  import MaterialIcon from './components/MaterialIcon.svelte'
  import { buildApiUrl, resolveApiBase } from './lib/apiBase'
  import { appendSessionActivity, openSession } from './lib/sessionApi'
  import { axisBuildersStore } from './lib/axisBuilderStore'
  import { axisSessionFromResponse } from './lib/axisSessions'
  import {
    deriveEffectiveSubsetIds,
    hasActiveSubset,
    normalizeHistogramSlices,
    normalizeSubsetFilters,
    stateFromSubsetChips,
    subsetChipsFromState,
  } from './lib/subsetState'

  const API_BASE = resolveApiBase()
  const SYSTEM_NAME = 'Reaxis'
  const SESSION_STORAGE_KEY = 'request.activeSession'

  let datasets = []
  let currentSession = ''
  let sessionPending = false
  let sessionError = ''
  let workspaceReady = false
  let datasetPath = ''
  let allImages = []
  let warningMsg = ''
  let labelDB = {}
  let axisBuilderSessions = []

  let axes = []
  let selectedAxisX = null
  let selectedAxisY = null
  let embedMethod = 'clip'
  let axisDebugAppliedEmbeddingOption = 'clip_raw'
  let axisDebugAppliedPriorMode = 'rank'
  let embedFileName = ''
  let histogramSlices = []
  let subsetFilters = []
  let minimapSizeOffset = 0
  let minimapFocusRequest = null
  let minimapViewState = null
  let minimapRestoreState = null
  let savedAxesOpen = false
  let savedAxes = []
  let savedAxesLoading = false
  let savedAxesSaving = false
  let savedAxesBusyId = ''
  let savedAxesError = ''
  let savedVisualizations = []
  let savedVisualizationsLoading = false
  let savedVisualizationsSaving = false
  let savedVisualizationsBusyId = ''
  let savedVisualizationsError = ''
  let savedSubsets = []
  let savedSubsetsLoading = false
  let savedSubsetsSaving = false
  let savedSubsetsBusyId = ''
  let savedSubsetsError = ''

  let minimapContainerRef
  let windowWidth = 0
  let windowHeight = 0
  let leftPanelWidth = 624
  let sidebarResize = null
  const MINIMAP_MIN_SIDE = 180

  const galleryCache = new Map()
  const prefetched = new Set()
  const MINIMAP_THUMB_SIZE = 64
  const PREFETCH_MAX = 320
  const unsubscribeAxisBuilders = axisBuildersStore.subscribe((value) => {
    axisBuilderSessions = Array.isArray(value) ? value : []
  })

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v))
  }

  function apiUrl(path) {
    return buildApiUrl(API_BASE, path)
  }

  $: normalizedHistogramSlices = normalizeHistogramSlices(histogramSlices)
  $: normalizedSubsetFilters = normalizeSubsetFilters(subsetFilters)
  $: subsetChips = subsetChipsFromState(normalizedHistogramSlices, normalizedSubsetFilters)
  $: subsetActive = hasActiveSubset(normalizedHistogramSlices, normalizedSubsetFilters)
  $: effectiveSubsetIds = deriveEffectiveSubsetIds(allImages, normalizedHistogramSlices, normalizedSubsetFilters)

  function axisDebugRequestForOption(rawOption) {
    const raw = String(rawOption || 'clip_raw').trim().toLowerCase()
    if (raw === 'clip') return { semanticMethod: 'clip', norm: true }
    if (raw === 'clip_raw') return { semanticMethod: 'clip', norm: false }
    if (raw === 'siglip2') return { semanticMethod: 'siglip2', norm: true }
    return { semanticMethod: 'siglip2', norm: false }
  }

  function axisDebugLayoutMethod(rawOption) {
    return String(axisDebugRequestForOption(rawOption)?.semanticMethod || 'clip').trim() || 'clip'
  }

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

  function sidebarWidthMax() {
    return clamp(Math.floor(windowWidth * 0.65), 468, 840)
  }

  function onSidebarResizeStart(e) {
    sidebarResize = {
      startX: Number(e.clientX || 0),
      startWidth: Number(leftPanelWidth || 480),
    }
    window.addEventListener('pointermove', onSidebarResizeMove)
    window.addEventListener('pointerup', onSidebarResizeEnd)
    try { e.preventDefault() } catch (_) {}
  }

  function onSidebarResizeMove(e) {
    if (!sidebarResize) return
    const dx = Number(e.clientX || 0) - sidebarResize.startX
    leftPanelWidth = clamp(sidebarResize.startWidth + dx, 468, sidebarWidthMax())
  }

  function onSidebarResizeEnd() {
    sidebarResize = null
    window.removeEventListener('pointermove', onSidebarResizeMove)
    window.removeEventListener('pointerup', onSidebarResizeEnd)
  }

  function persistSessionName(sessionName) {
    if (typeof window === 'undefined') return
    try {
      if (sessionName) window.localStorage.setItem(SESSION_STORAGE_KEY, sessionName)
      else window.localStorage.removeItem(SESSION_STORAGE_KEY)
    } catch (_) {}
  }

  async function logSessionAction(action, detail) {
    if (!currentSession) return
    try {
      await appendSessionActivity(API_BASE, currentSession, action, detail)
    } catch (_) {}
  }

  function queueSessionAction(action, detail) {
    logSessionAction(action, detail)
  }

  function resetSessionScopedUiState() {
    labelDB = {}
    histogramSlices = []
    subsetFilters = []
    minimapViewState = null
    minimapRestoreState = null
    savedAxes = []
    savedVisualizations = []
    savedSubsets = []
    savedAxesBusyId = ''
    savedVisualizationsBusyId = ''
    savedSubsetsBusyId = ''
    savedAxesError = ''
    savedVisualizationsError = ''
    savedSubsetsError = ''
    savedAxesOpen = false
    axisBuildersStore.reset()
    axes = (axes || []).filter((axis) => {
      const group = String(axis?.group || '')
      return group === 'base' || group === 'meta'
    })
    selectedAxisX = fallbackAxisId('x', axes)
    selectedAxisY = fallbackAxisId('y', axes)
  }

  async function ensureWorkspaceLoaded() {
    if (workspaceReady) return
    await loadDatasets()
    await loadGallery()
    workspaceReady = true
  }

  async function activateSession(rawSessionName) {
    const requested = String(rawSessionName || '').trim()
    if (!requested) {
      sessionError = 'Enter a session name first.'
      return
    }
    sessionPending = true
    sessionError = ''
    try {
      const data = await openSession(API_BASE, requested)
      const nextSession = String(data?.session || data?.item?.id || '').trim()
      if (!nextSession) throw new Error('Backend did not return a session name.')
      const changed = nextSession !== currentSession
      currentSession = nextSession
      persistSessionName(nextSession)
      await ensureWorkspaceLoaded()
      if (changed) resetSessionScopedUiState()
      await reloadSavedLibraries()
    } catch (err) {
      currentSession = ''
      persistSessionName('')
      sessionError = `Failed to open session: ${String(err)}`
    } finally {
      sessionPending = false
    }
  }

  function closeSessionGate() {
    currentSession = ''
    sessionError = ''
    sessionPending = false
    savedAxesLoading = false
    savedAxesSaving = false
    savedVisualizationsLoading = false
    savedVisualizationsSaving = false
    savedSubsetsLoading = false
    savedSubsetsSaving = false
    persistSessionName('')
    resetSessionScopedUiState()
  }

  async function loadSavedAxesLibrary() {
    if (!currentSession) {
      savedAxes = []
      savedAxesLoading = false
      return
    }
    savedAxesLoading = true
    savedAxesError = ''
    try {
      const qs = new URLSearchParams()
      qs.set('session', currentSession)
      const res = await fetch(apiUrl(`/axis/library?${qs.toString()}`))
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      savedAxes = Array.isArray(data?.items) ? data.items : []
    } catch (e) {
      savedAxes = []
      savedAxesError = `Failed to load saved axes: ${String(e)}`
    } finally {
      savedAxesLoading = false
    }
  }

  async function loadSavedVisualizationsLibrary() {
    if (!currentSession) {
      savedVisualizations = []
      savedVisualizationsLoading = false
      return
    }
    savedVisualizationsLoading = true
    savedVisualizationsError = ''
    try {
      const qs = new URLSearchParams()
      qs.set('session', currentSession)
      const res = await fetch(apiUrl(`/visualization/library?${qs.toString()}`))
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      savedVisualizations = Array.isArray(data?.items) ? data.items : []
    } catch (e) {
      savedVisualizations = []
      savedVisualizationsError = `Failed to load saved visualizations: ${String(e)}`
    } finally {
      savedVisualizationsLoading = false
    }
  }

  async function loadSavedSubsetsLibrary() {
    if (!currentSession) {
      savedSubsets = []
      savedSubsetsLoading = false
      return
    }
    savedSubsetsLoading = true
    savedSubsetsError = ''
    try {
      const qs = new URLSearchParams()
      qs.set('session', currentSession)
      const res = await fetch(apiUrl(`/subset/library?${qs.toString()}`))
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      savedSubsets = Array.isArray(data?.items) ? data.items : []
    } catch (e) {
      savedSubsets = []
      savedSubsetsError = `Failed to load saved subsets: ${String(e)}`
    } finally {
      savedSubsetsLoading = false
    }
  }

  async function reloadSavedLibraries() {
    await Promise.all([
      loadSavedAxesLibrary(),
      loadSavedVisualizationsLibrary(),
      loadSavedSubsetsLibrary(),
    ])
  }

  async function saveCurrentSubset() {
    if (!currentSession || !datasetPath || !subsetActive) return
    const defaultName = `${datasetPath} · ${effectiveSubsetIds.length} imgs`
    const entered = typeof window !== 'undefined' ? window.prompt('Save subset as', defaultName) : defaultName
    const name = String(entered || '').trim()
    if (!name) return

    savedSubsetsSaving = true
    savedSubsetsError = ''
    try {
      const data = await postJson('/subset/library/save', {
        session: currentSession,
        name,
        dataset: datasetPath || undefined,
        subset_chips: subsetChips,
        image_count: effectiveSubsetIds.length,
      })
      savedSubsets = Array.isArray(data?.items) ? data.items : savedSubsets
      savedAxesOpen = true
      await logSessionAction('save subset', name)
    } catch (err) {
      savedSubsetsError = `Save failed: ${String(err)}`
    } finally {
      savedSubsetsSaving = false
    }
  }

  async function saveAxisToLibrary(e) {
    if (!currentSession) return
    const axisId = String(e?.detail?.axisId || '').trim()
    const axisName = String(e?.detail?.axisName || '').trim()
    const q = String(e?.detail?.q || axisName).trim()
    const modelType = String(e?.detail?.modelType || '').trim()
    if (!axisId || !axisName || !q) return
    savedAxesSaving = true
    savedAxesError = ''
    try {
      const data = await postJson('/axis/library/save', {
        session: currentSession,
        axis_id: axisId,
        name: axisName,
        q,
        dataset: datasetPath || undefined,
        origin_dataset: datasetPath || undefined,
        model_type: modelType || undefined,
      })
      savedAxes = Array.isArray(data?.items) ? data.items : savedAxes
      savedAxesOpen = true
      await logSessionAction('save ax', axisName)
    } catch (err) {
      savedAxesError = `Save failed: ${String(err)}`
    } finally {
      savedAxesSaving = false
    }
  }

  async function projectSavedSubset(item) {
    if (!currentSession || !item?.id) return
    savedSubsetsBusyId = String(item.id)
    savedSubsetsError = ''
    try {
      const data = await postJson('/subset/library/project', {
        session: currentSession,
        subset_id: item.id,
      })
      const subset = data?.subset || {}
      const targetDataset = String(subset?.dataset || '').trim() || datasetPath
      if (targetDataset && targetDataset !== datasetPath) {
        datasetPath = targetDataset
        resetStateForDatasetChange()
        await loadGallery()
      } else if ((allImages || []).length === 0 && targetDataset) {
        await loadGallery()
      }
      const restored = stateFromSubsetChips(Array.isArray(subset?.subset_chips) ? subset.subset_chips : [])
      histogramSlices = restored.histogramSlices
      subsetFilters = restored.subsetFilters
      savedAxesOpen = false
      await logSessionAction('retrieve subset', item?.name || item?.id || 'none')
    } catch (err) {
      savedSubsetsError = `Load failed: ${String(err)}`
    } finally {
      savedSubsetsBusyId = ''
    }
  }

  async function removeSavedSubset(itemId) {
    if (!currentSession) return
    const id = String(itemId || '').trim()
    if (!id) return
    savedSubsetsBusyId = id
    savedSubsetsError = ''
    try {
      const qs = new URLSearchParams()
      qs.set('session', currentSession)
      const res = await fetch(apiUrl(`/subset/library/${encodeURIComponent(id)}?${qs.toString()}`), { method: 'DELETE' })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      savedSubsets = Array.isArray(data?.items) ? data.items : savedSubsets.filter((it) => String(it?.id || '') !== id)
    } catch (err) {
      savedSubsetsError = `Delete failed: ${String(err)}`
    } finally {
      savedSubsetsBusyId = ''
    }
  }

  async function projectSavedAxis(item) {
    if (!currentSession) return
    if (!item?.id) return
    savedAxesBusyId = String(item.id)
    savedAxesError = ''
    try {
      const data = await postJson('/axis/library/project', {
        session: currentSession,
        library_axis_id: item.id,
        dataset: datasetPath || undefined,
        collection_id: datasetPath || undefined,
      })
      const session = axisSessionFromResponse({ q: item.name, axis: { name: item.name } }, data, { preferredName: item.name })
      if (session?.axis?.id) {
        upsertAxisValue({ ...session.axis, group: session.axis.group || 'prompt' })
        axisBuildersStore.upsert(session)
        await logSessionAction('retrieve axis', item?.name || item?.q || item?.id || 'none')
      }
    } catch (err) {
      savedAxesError = `Load failed: ${String(err)}`
    } finally {
      savedAxesBusyId = ''
    }
  }

  async function removeSavedAxis(itemId) {
    if (!currentSession) return
    const id = String(itemId || '').trim()
    if (!id) return
    savedAxesBusyId = id
    savedAxesError = ''
    try {
      const qs = new URLSearchParams()
      qs.set('session', currentSession)
      const res = await fetch(apiUrl(`/axis/library/${encodeURIComponent(id)}?${qs.toString()}`), { method: 'DELETE' })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      savedAxes = Array.isArray(data?.items) ? data.items : savedAxes.filter((it) => String(it?.id || '') !== id)
    } catch (err) {
      savedAxesError = `Delete failed: ${String(err)}`
    } finally {
      savedAxesBusyId = ''
    }
  }

  function axisNameById(id) {
    const key = String(id || '').trim()
    return (axes || []).find((axis) => String(axis?.id || '').trim() === key)?.name || ''
  }

  function describeViewSelection(xId = selectedAxisX, yId = selectedAxisY) {
    const xName = axisNameById(xId) || 'none'
    const yName = axisNameById(yId) || 'none'
    return `x:${xName}, y:${yName}`
  }

  function applyUserAxisSelection(nextX, nextY) {
    selectedAxisX = String(nextX || '').trim() || null
    selectedAxisY = String(nextY || '').trim() || null
    queueSessionAction('change view', describeViewSelection(selectedAxisX, selectedAxisY))
  }

  function applyUserAxisSelectionSlot(slot, nextId) {
    const id = String(nextId || '').trim() || null
    if (slot === 'x') applyUserAxisSelection(id, selectedAxisY)
    else applyUserAxisSelection(selectedAxisX, id)
  }

  function customAxesForSnapshot() {
    const builderIds = new Set((axisBuilderSessions || []).map((entry) => String(entry?.axisId || '').trim()).filter(Boolean))
    return (axes || [])
      .filter((axis) => builderIds.has(String(axis?.id || '').trim()))
      .map((axis) => ({
        id: String(axis?.id || '').trim(),
        name: String(axis?.name || '').trim(),
        group: String(axis?.group || 'prompt').trim(),
      }))
  }

  async function saveCurrentVisualization() {
    if (!currentSession) return
    if (!datasetPath) return
    const defaultNameParts = [datasetPath]
    if (selectedAxisX) defaultNameParts.push(axisNameById(selectedAxisX) || 'X')
    if (selectedAxisY) defaultNameParts.push(axisNameById(selectedAxisY) || 'Y')
    const defaultName = defaultNameParts.join(' · ')
    const entered = typeof window !== 'undefined' ? window.prompt('Save visualization as', defaultName) : defaultName
    const name = String(entered || '').trim()
    if (!name) return

    const customAxes = customAxesForSnapshot()
    savedVisualizationsSaving = true
    savedVisualizationsError = ''
    try {
      const data = await postJson('/visualization/library/save', {
        session: currentSession,
        name,
        dataset: datasetPath || undefined,
        selected_x: selectedAxisX || '',
        selected_y: selectedAxisY || '',
        selected_x_name: axisNameById(selectedAxisX),
        selected_y_name: axisNameById(selectedAxisY),
        histogram_slices: normalizedHistogramSlices || [],
        subset_chips: subsetChips || [],
        subset_filters: normalizedSubsetFilters || [],
        subset_filter: null,
        view_state: minimapViewState || {},
        minimap_size_offset: minimapSizeOffset || 0,
        axis_ids: customAxes.map((axis) => axis.id),
        axes_manifest: customAxes,
      })
      savedVisualizations = Array.isArray(data?.items) ? data.items : savedVisualizations
      savedAxesOpen = true
      await logSessionAction('save visualization', name)
    } catch (err) {
      savedVisualizationsError = `Save failed: ${String(err)}`
    } finally {
      savedVisualizationsSaving = false
    }
  }

  function clearCustomAxesAndBuilders() {
    axes = (axes || []).filter((axis) => {
      const group = String(axis?.group || '')
      return group === 'base' || group === 'meta'
    })
    axisBuildersStore.reset()
    histogramSlices = []
  }

  function mapSavedAxisId(oldId, axisIdMap) {
    const key = String(oldId || '').trim()
    if (!key) return null
    return String(axisIdMap?.[key] || key).trim() || null
  }

  async function projectSavedVisualization(item) {
    if (!currentSession) return
    if (!item?.id) return
    savedVisualizationsBusyId = String(item.id)
    savedVisualizationsError = ''
    try {
      const data = await postJson('/visualization/library/project', {
        session: currentSession,
        visualization_id: item.id,
      })
      const visualization = data?.visualization || {}
      const targetDataset = String(visualization?.dataset || '').trim() || datasetPath
      if (targetDataset && targetDataset !== datasetPath) {
        datasetPath = targetDataset
        resetStateForDatasetChange()
        await loadGallery()
      } else if ((allImages || []).length === 0 && targetDataset) {
        await loadGallery()
      }

      clearCustomAxesAndBuilders()

      const projectedAxes = Array.isArray(data?.projected_axes) ? data.projected_axes : []
      for (const payload of projectedAxes) {
        const nextSession = axisSessionFromResponse({ q: payload?.q || payload?.axis?.name, axis: { name: payload?.axis?.name } }, payload, {
          preferredName: payload?.axis?.name,
        })
        if (nextSession?.axis?.id) {
          upsertAxisValue({ ...nextSession.axis, group: nextSession.axis.group || 'prompt' })
          axisBuildersStore.upsert(nextSession)
        }
      }

      const axisIdMap = data?.axis_id_map || {}
      selectedAxisX = mapSavedAxisId(visualization?.selected_x, axisIdMap)
      selectedAxisY = mapSavedAxisId(visualization?.selected_y, axisIdMap)
      if (selectedAxisX && !(axes || []).some((axis) => axis?.id === selectedAxisX)) {
        selectedAxisX = fallbackAxisId('x', axes)
      }
      if (selectedAxisY && !(axes || []).some((axis) => axis?.id === selectedAxisY)) {
        selectedAxisY = fallbackAxisId('y', axes)
      }
      const restoredSubsetState = (() => {
        const subsetChipsPayload = Array.isArray(visualization?.subset_chips) ? visualization.subset_chips : []
        if (subsetChipsPayload.length > 0) {
          const restored = stateFromSubsetChips(subsetChipsPayload)
          return {
            histogramSlices: restored.histogramSlices
              .map((slice) => {
                const axisId = mapSavedAxisId(slice.axisId, axisIdMap)
                return axisId ? { ...slice, axisId } : slice
              }),
            subsetFilters: restored.subsetFilters,
          }
        }
        const legacySlices = (Array.isArray(visualization?.histogram_slices) ? visualization.histogram_slices : [])
          .map((slice) => {
            if (!slice || typeof slice !== 'object') return null
            const axisId = mapSavedAxisId(slice.axisId, axisIdMap)
            return axisId ? { ...slice, axisId } : null
          })
          .filter(Boolean)
        const legacyFilters = Array.isArray(visualization?.subset_filters)
          ? visualization.subset_filters
          : (visualization?.subset_filter ? [visualization.subset_filter] : [])
        return {
          histogramSlices: legacySlices,
          subsetFilters: legacyFilters,
        }
      })()
      histogramSlices = restoredSubsetState.histogramSlices
      subsetFilters = restoredSubsetState.subsetFilters
      minimapSizeOffset = Number(visualization?.minimap_size_offset || 0) || 0
      minimapRestoreState = {
        ...(visualization?.view_state && typeof visualization.view_state === 'object' ? visualization.view_state : {}),
        nonce: Date.now(),
      }
      savedAxesOpen = false
      await logSessionAction('retrieve visualization', item?.name || item?.id || 'none')
    } catch (err) {
      savedVisualizationsError = `Load failed: ${String(err)}`
    } finally {
      savedVisualizationsBusyId = ''
    }
  }

  async function removeSavedVisualization(itemId) {
    if (!currentSession) return
    const id = String(itemId || '').trim()
    if (!id) return
    savedVisualizationsBusyId = id
    savedVisualizationsError = ''
    try {
      const qs = new URLSearchParams()
      qs.set('session', currentSession)
      const res = await fetch(apiUrl(`/visualization/library/${encodeURIComponent(id)}?${qs.toString()}`), { method: 'DELETE' })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      savedVisualizations = Array.isArray(data?.items) ? data.items : savedVisualizations.filter((it) => String(it?.id || '') !== id)
    } catch (err) {
      savedVisualizationsError = `Delete failed: ${String(err)}`
    } finally {
      savedVisualizationsBusyId = ''
    }
  }

  $: panelHeight = Math.max(320, windowHeight - 120 - (warningMsg ? 56 : 0))
  $: minimapAvailWidth = (() => {
    void windowWidth
    return minimapContainerRef ? Math.floor(minimapContainerRef.clientWidth) : Math.max(320, windowWidth - leftPanelWidth - 84)
  })()
  $: minimapAvailHeight = (() => {
    void windowHeight
    return minimapContainerRef ? Math.floor(minimapContainerRef.clientHeight) : panelHeight
  })()
  $: minimapMaxSide = Math.max(MINIMAP_MIN_SIDE, Math.min(minimapAvailWidth, minimapAvailHeight))
  $: minimapSide = clamp(minimapMaxSide + minimapSizeOffset, MINIMAP_MIN_SIDE, minimapMaxSide)
  $: minimapItems = (allImages || []).map((item) => ({
    id: item.id,
    url: item.url,
    thumbUrl: item.thumbUrl,
    fullUrl: item.url,
    gx: item.gx,
    gy: item.gy,
    x: item.x,
    y: item.y,
    embed: item.embed,
  }))

  function resetStateForDatasetChange() {
    allImages = []
    axes = []
    selectedAxisX = null
    selectedAxisY = null
    embedMethod = axisDebugLayoutMethod(axisDebugAppliedEmbeddingOption)
    labelDB = {}
    histogramSlices = []
    subsetFilters = []
    minimapViewState = null
    minimapRestoreState = null
    minimapSizeOffset = 0
    warningMsg = ''
    galleryCache.clear()
    axisBuildersStore.reset()
  }

  function isBaseAxisId(axisId) {
    const id = String(axisId || '').trim()
    if (!id) return false
    if (/^axis:[^:]+:[xy]$/.test(id)) return true
    const axis = (axes || []).find((entry) => String(entry?.id || '').trim() === id)
    return String(axis?.group || '').trim() === 'base'
  }

  function ensureDefaultAxesForCurrentProjection() {
    const methodName = String(embedMethod || 'clip')
    const idX = `axis:${methodName}:x`
    const idY = `axis:${methodName}:y`
    const previousXWasBase = isBaseAxisId(selectedAxisX)
    const previousYWasBase = isBaseAxisId(selectedAxisY)

    const coordsX = {}
    const coordsY = {}
    for (const item of allImages) {
      coordsX[item.id] = Number(item.x ?? 0)
      coordsY[item.id] = Number(item.y ?? 0)
    }

    const next = new Map(
      (axes || [])
        .filter((axis) => String(axis?.group || '').trim() !== 'base')
        .map((axis) => [axis.id, axis])
    )
    next.set(idX, { id: idX, name: 'Embedding X', coords: coordsX, group: 'base' })
    next.set(idY, { id: idY, name: 'Embedding Y', coords: coordsY, group: 'base' })
    axes = Array.from(next.values())

    if (!selectedAxisX || previousXWasBase || !next.has(selectedAxisX)) selectedAxisX = idX
    if (!selectedAxisY || previousYWasBase || !next.has(selectedAxisY)) selectedAxisY = idY
  }

  function toThumbPath(u, size = MINIMAP_THUMB_SIZE) {
    if (!u) return ''
    if (u.startsWith('/images/')) return `/thumb/${Math.max(16, Math.min(1024, Number(size) || MINIMAP_THUMB_SIZE))}${u.substring('/images'.length)}`
    return u
  }

  function prefixUrl(u) {
    if (!u) return ''
    if (u.startsWith('http://') || u.startsWith('https://')) return u
    if (u.startsWith('/')) return buildApiUrl(API_BASE, u)
    return u
  }

  function thumbUrl(u, size = MINIMAP_THUMB_SIZE) {
    return prefixUrl(toThumbPath(u, size))
  }

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

  async function loadDatasets() {
    try {
      const res = await fetch(apiUrl('/datasets'))
      if (!res.ok) throw new Error(`datasets ${res.status}`)
      const data = await res.json()
      if (!Array.isArray(data) || data.length === 0) throw new Error('empty datasets')
      datasets = data.map((d) => ({
        label: d?.label || d?.value || 'Dataset',
        value: d?.value || d?.label || '',
      }))
      if (!datasetPath && datasets.length > 0) {
        const preferredDataset = datasets.find((item) => String(item?.value || '').trim() === 'EmoSet')
        datasetPath = preferredDataset?.value || datasets[0].value
      }
      return
    } catch (e) {
      datasets = []
      datasetPath = ''
      warningMsg = `Failed to load datasets from backend: ${String(e)}`
    }
  }

  async function loadGallery() {
    const cacheKey = `${datasetPath}|${embedMethod}|initial`
    if (galleryCache.has(cacheKey)) {
      const cached = galleryCache.get(cacheKey)
      allImages = cached.items
      warningMsg = cached.warning || ''
      embedFileName = String(cached.embeddingFile || '').trim()
      embedMethod = String(cached.embedMethod || embedMethod || 'clip').trim() || 'clip'
      ensureDefaultAxesForCurrentProjection()
      return true
    }

    const qs = new URLSearchParams()
    qs.set('embed', embedMethod)
    if (datasetPath) qs.set('dataset', datasetPath)

    try {
      const res = await fetch(apiUrl(`/gallery.json?${qs.toString()}`))
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      if (!data || !Array.isArray(data.items)) throw new Error('Invalid gallery payload')
      const effectiveEmbedMethod = String(data.embed || embedMethod || 'clip')
      if (effectiveEmbedMethod) embedMethod = effectiveEmbedMethod
      embedFileName = String(data.embedding_file || '').trim()

      allImages = data.items.map((item) => ({
        ...(() => {
          const base = item.url || item.path || ''
          return {
            url: prefixUrl(base),
            thumbUrl: thumbUrl(base, MINIMAP_THUMB_SIZE),
          }
        })(),
        id: item.id,
        className: item.className || 'Unknown',
        label: item.id,
        gx: Number(item.gx ?? item.x ?? 0),
        gy: Number(item.gy ?? item.y ?? 0),
        x: Number(item.x ?? item.gx ?? 0),
        y: Number(item.y ?? item.gy ?? 0),
        embed: Array.isArray(item.embed)
          ? item.embed.map((v) => Number(v)).filter((v) => Number.isFinite(v))
          : [Number(item.x ?? item.gx ?? 0), Number(item.y ?? item.gy ?? 0)],
      }))
      warningMsg = data.warning || ''

      const incomingMeta = Array.isArray(data.metadata_axes) ? data.metadata_axes : []
      if (incomingMeta.length > 0) {
        const existing = new Map(axes.map((axis) => [axis.id, axis]))
        for (const axis of incomingMeta) {
          if (!axis?.id || existing.has(axis.id)) continue
          existing.set(axis.id, {
            id: axis.id,
            name: axis.name || axis.id,
            coords: axis.coords || {},
            labels: axis.labels || [],
            label_positions: axis.label_positions || [],
            group: 'meta',
          })
        }
        axes = Array.from(existing.values())
      }

      galleryCache.set(cacheKey, {
        items: allImages,
        warning: warningMsg,
        embeddingFile: embedFileName,
        embedMethod: String(embedMethod || effectiveEmbedMethod || 'clip').trim() || 'clip',
      })
      schedulePrefetch(allImages.slice(0, PREFETCH_MAX).map((item) => item.thumbUrl || item.url))
      ensureDefaultAxesForCurrentProjection()
      return true
    } catch (e) {
      warningMsg = `Failed to load gallery from backend: ${String(e)}`
      allImages = []
      embedFileName = ''
      ensureDefaultAxesForCurrentProjection()
      return false
    }
  }

  onMount(async () => {
    const stored = typeof window !== 'undefined'
      ? String(window.localStorage.getItem(SESSION_STORAGE_KEY) || '').trim()
      : ''
    if (stored) await activateSession(stored)
  })

  onDestroy(() => {
    onSidebarResizeEnd()
    unsubscribeAxisBuilders()
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

  function cloneAxisValue(axis, existing = null) {
    const source = axis && typeof axis === 'object' ? axis : {}
    const previous = existing && typeof existing === 'object' ? existing : {}
    return {
      ...source,
      group: String(source.group || previous.group || 'prompt').trim() || 'prompt',
      coords: source.coords && typeof source.coords === 'object' ? { ...source.coords } : {},
      labels: Array.isArray(source.labels) ? [...source.labels] : [],
      label_positions: Array.isArray(source.label_positions) ? [...source.label_positions] : [],
    }
  }

  function upsertAxisValue(axis) {
    if (!axis || !axis.id) return
    let found = false
    const next = axes.map((existing) => {
      if (existing.id !== axis.id) return existing
      found = true
      return cloneAxisValue(axis, existing)
    })
    axes = found ? next : [cloneAxisValue(axis), ...next]
  }

  function upsertAxis(e) {
    upsertAxisValue(e.detail?.axis)
  }

  function fallbackAxisId(slot, nextAxes) {
    const preferred = `axis:${String(embedMethod || 'clip')}:${slot}`
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
    if (Array.isArray(histogramSlices) && histogramSlices.length > 0) {
      histogramSlices = histogramSlices.filter((slice) => String(slice?.axisId || '').trim() !== id)
    }
  }

  function onRemoveAxis(e) {
    removeAxisValue(e.detail?.id)
  }
</script>

<svelte:window bind:innerWidth={windowWidth} bind:innerHeight={windowHeight} />

{#if currentSession}
<div class="app-main app-shell text-gray-900">
  <header class="app-header">
    <div class="app-header-inner">
      <div class="i-heroicons-sparkles brand-icon" />
      <div class="brand-name">Reaxis</div>
      <div class="inline-flex items-center gap-2">
        <label for="dataset-select" class="text-xs text-gray-700">Dataset</label>
        <select
          id="dataset-select"
          class="text-xs"
          on:change={async (e) => {
            const next = e.currentTarget.value
            if ((datasetPath || '') === (next || '')) return
            datasetPath = next
            resetStateForDatasetChange()
            const loaded = await loadGallery()
            if (loaded) await logSessionAction('change dataset', next)
          }}
        >
          {#each datasets as d}
            <option value={d.value} selected={(datasetPath || '') === (d.value || '')}>{d.label}</option>
          {/each}
        </select>
      </div>
      <div class="session-chip" title={`Active session: ${currentSession}`}>
        <span class="session-chip-label">Session</span>
        <span class="session-chip-value">{currentSession}</span>
        <button
          type="button"
          class="session-chip-close"
          aria-label="Switch session"
          title="Switch session"
          on:click={closeSessionGate}
        ><MaterialIcon name="lock" size={14} /></button>
      </div>
      <div class="flex-1" />
      <button
        type="button"
        class="btn btn-icon btn-ui-secondary saved-axes-toggle"
        aria-label="Open saved library"
        title="Saved library"
        on:click={() => { savedAxesOpen = !savedAxesOpen }}
      ><MaterialIcon name="menu" /></button>
    </div>
  </header>

  {#if warningMsg}
    <div class="bg-yellow-50 border-l-4 border-yellow-400 text-yellow-800 p-3">
      <div class="text-sm px-4">{warningMsg}</div>
    </div>
  {/if}

  <main class="app-main app-workspace w-full py-0">
    <div class="workspace-grid px-0">
      <aside class="workspace-sidebar shrink-0" style={`width:${leftPanelWidth}px;min-width:468px;`}>
        <div class="tile tile-primary">
          <div class="tile-header flex items-center gap-2">
            <span class="i-heroicons-chat-bubble-left-right text-slate-600" />
            Axis suggestion
          </div>
          <div class="tile-content">
            <PromptSidebar
              {axes}
              items={allImages}
              externalSlices={normalizedHistogramSlices}
              subsetIds={effectiveSubsetIds}
              subsetActive={subsetActive}
              selectedX={selectedAxisX}
              selectedY={selectedAxisY}
              apiBase={API_BASE}
              sessionName={currentSession}
              dataset={datasetPath}
              axisDebugEmbeddingOption={axisDebugAppliedEmbeddingOption}
              axisDebugPriorMode={axisDebugAppliedPriorMode}
              on:upsertAxis={upsertAxis}
              on:logAction={(e) => queueSessionAction(e.detail?.action, e.detail?.detail)}
              on:removeAxis={onRemoveAxis}
              on:setX={(e) => { if (e.detail?.id) applyUserAxisSelectionSlot('x', e.detail.id) }}
              on:setY={(e) => { if (e.detail?.id) applyUserAxisSelectionSlot('y', e.detail.id) }}
              on:sliceChange={(e) => {
                const many = Array.isArray(e.detail?.slices) ? e.detail.slices : null
                const single = e.detail?.slice
                histogramSlices = many || (single ? [single] : [])
              }}
              on:openImage={(e) => {
                const imageId = String(e.detail?.imageId || '').trim()
                if (!imageId) return
                minimapFocusRequest = { imageId, nonce: Date.now() }
              }}
              on:saveAxis={saveAxisToLibrary}
            />
          </div>
        </div>
      </aside>

      <div
        class="sidebar-resizer"
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize axes sidebar"
        title="Drag to resize sidebar"
        on:pointerdown={onSidebarResizeStart}
      />

      <section class="workspace-center min-w-0 flex-1">
        <div class="tile tile-primary scatter-workspace-tile">
          <div class="tile-header flex items-center gap-2">
            <span class="i-heroicons-chart-bar-square text-slate-600" />
            Visualization
          </div>
          <div class="tile-content flush minimap-panel" bind:this={minimapContainerRef}>
            <AxesMinimap
              items={minimapItems}
              apiBase={API_BASE}
              sessionName={currentSession}
              selections={[]}
              axes={axes}
              width={minimapSide}
              height={minimapSide}
              focusImageRequest={minimapFocusRequest}
              labels={new Map(Object.entries(labelDB))}
              histogramSlices={normalizedHistogramSlices}
              subsetFilters={normalizedSubsetFilters}
              saveVisualizationDisabled={!datasetPath || savedVisualizationsSaving}
              selectionToolsEnabled={false}
              restoreViewState={minimapRestoreState}
              bind:selectedX={selectedAxisX}
              bind:selectedY={selectedAxisY}
              on:logAction={(e) => queueSessionAction(e.detail?.action, e.detail?.detail)}
              on:saveVisualization={saveCurrentVisualization}
              on:sliceChange={(e) => {
                const many = Array.isArray(e.detail?.slices) ? e.detail.slices : null
                const single = e.detail?.slice
                histogramSlices = many || (single ? [single] : [])
              }}
              on:viewStateChange={(e) => {
                minimapViewState = e.detail || {}
              }}
              on:subsetFiltersChange={(e) => {
                subsetFilters = Array.isArray(e.detail?.filters) ? e.detail.filters : []
              }}
              on:saveSubset={saveCurrentSubset}
              on:axesChange={(e) => {
                selectedAxisX = String(e.detail?.selectedX || '').trim() || null
                selectedAxisY = String(e.detail?.selectedY || '').trim() || null
              }}
              on:label={onScribbleLabel}
              on:create={(e) => {
                const axis = e.detail
                if (!axis || !axis.id) return
                upsertAxisValue({ ...axis, group: axis.group || 'prompt' })
              }}
              on:resizeMinimap={(e) => {
                const deltaPx = Number(e.detail?.deltaPx || 0)
                if (!Number.isFinite(deltaPx) || Math.abs(deltaPx) < 0.001) return
                minimapSizeOffset = clamp(minimapSizeOffset + deltaPx, -560, 1400)
              }}
            />
          </div>
        </div>
      </section>
    </div>
  </main>

  {#if savedAxesOpen}
    <button
      class="saved-axes-scrim"
      aria-label="Close saved axes sidebar"
      on:click={() => { savedAxesOpen = false }}
    />
  {/if}
  <aside class={`saved-axes-drawer ${savedAxesOpen ? 'open' : ''}`} aria-hidden={!savedAxesOpen}>
    <div class="saved-axes-header">
      <div class="saved-axes-title">Saved library</div>
      <button type="button" class="saved-axes-close" aria-label="Close saved axes sidebar" on:click={() => { savedAxesOpen = false }}><MaterialIcon name="close" /></button>
    </div>
    <div class="saved-library-sections">
      <section class="saved-library-section">
        <div class="saved-library-section-header">
          <div class="saved-library-section-title">Saved axes</div>
          <div class="saved-library-section-meta">{savedAxesSaving ? 'Saving...' : `${savedAxes.length}`}</div>
        </div>
        {#if savedAxesError}
          <div class="saved-axes-error">{savedAxesError}</div>
        {/if}
        <div class="saved-axes-body">
          {#if savedAxesLoading}
            <div class="saved-axes-empty">Loading...</div>
          {:else if savedAxes.length === 0}
            <div class="saved-axes-empty">No saved axes</div>
          {:else}
            {#each savedAxes as item (item.id)}
              <div class="saved-axis-row">
                <button
                  type="button"
                  class="saved-axis-load"
                  disabled={savedAxesBusyId === item.id}
                  on:click={() => projectSavedAxis(item)}
                >
                  <div class="saved-axis-name">{item.name || item.q || item.id}</div>
                  <div class="saved-axis-origin">{item.origin_dataset || 'unknown'}</div>
                </button>
                <button
                  type="button"
                  class="saved-axis-delete"
                  aria-label="Remove saved axis"
                  title="Remove saved axis"
                  disabled={savedAxesBusyId === item.id}
                  on:click|stopPropagation={() => removeSavedAxis(item.id)}
                ><MaterialIcon name="close" /></button>
              </div>
            {/each}
          {/if}
        </div>
      </section>
      <section class="saved-library-section">
        <div class="saved-library-section-header">
          <div class="saved-library-section-title">Saved visualizations</div>
          <div class="saved-library-section-meta">{savedVisualizationsSaving ? 'Saving...' : `${savedVisualizations.length}`}</div>
        </div>
        {#if savedVisualizationsError}
          <div class="saved-axes-error">{savedVisualizationsError}</div>
        {/if}
        <div class="saved-axes-body">
          {#if savedVisualizationsLoading}
            <div class="saved-axes-empty">Loading...</div>
          {:else if savedVisualizations.length === 0}
            <div class="saved-axes-empty">No saved visualizations</div>
          {:else}
            {#each savedVisualizations as item (item.id)}
              <div class="saved-axis-row">
                <button
                  type="button"
                  class="saved-axis-load"
                  disabled={savedVisualizationsBusyId === item.id}
                  on:click={() => projectSavedVisualization(item)}
                >
                  <div class="saved-axis-name">{item.name || item.id}</div>
                  <div class="saved-axis-origin">{item.dataset || 'unknown'}</div>
                </button>
                <button
                  type="button"
                  class="saved-axis-delete"
                  aria-label="Remove saved visualization"
                  title="Remove saved visualization"
                  disabled={savedVisualizationsBusyId === item.id}
                  on:click|stopPropagation={() => removeSavedVisualization(item.id)}
                ><MaterialIcon name="close" /></button>
              </div>
            {/each}
          {/if}
        </div>
      </section>
      <section class="saved-library-section">
        <div class="saved-library-section-header">
          <div class="saved-library-section-title">Saved subsets</div>
          <div class="saved-library-section-meta">{savedSubsetsSaving ? 'Saving...' : `${savedSubsets.length}`}</div>
        </div>
        {#if savedSubsetsError}
          <div class="saved-axes-error">{savedSubsetsError}</div>
        {/if}
        <div class="saved-axes-body">
          {#if savedSubsetsLoading}
            <div class="saved-axes-empty">Loading...</div>
          {:else if savedSubsets.length === 0}
            <div class="saved-axes-empty">No saved subsets</div>
          {:else}
            {#each savedSubsets as item (item.id)}
              <div class="saved-axis-row">
                <button
                  type="button"
                  class="saved-axis-load"
                  disabled={savedSubsetsBusyId === item.id}
                  on:click={() => projectSavedSubset(item)}
                >
                  <div class="saved-axis-name">{item.name || item.id}</div>
                  <div class="saved-axis-origin">{item.dataset || 'unknown'} · {item.image_count || 0} imgs</div>
                </button>
                <button
                  type="button"
                  class="saved-axis-delete"
                  aria-label="Remove saved subset"
                  title="Remove saved subset"
                  disabled={savedSubsetsBusyId === item.id}
                  on:click|stopPropagation={() => removeSavedSubset(item.id)}
                ><MaterialIcon name="close" /></button>
              </div>
            {/each}
          {/if}
        </div>
      </section>
    </div>
  </aside>
</div>
{:else}
  <LoadingGate
    systemName={SYSTEM_NAME}
    busy={sessionPending}
    error={sessionError}
    on:submit={(e) => activateSession(e.detail?.sessionName)}
  />
{/if}

<style>
  :global(html, body, #app) { height: 100%; }

  .app-shell {
    height: 100vh;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .app-workspace {
    flex: 1 1 auto;
    min-height: 0;
    overflow: hidden;
    padding-top: 0;
    padding-bottom: 0;
  }

  .scatter-workspace-tile {
    height: 100%;
    display: flex;
    flex-direction: column;
  }

  .workspace-grid {
    display: flex;
    flex-wrap: nowrap;
    gap: 0;
    align-items: stretch;
    height: 100%;
    overflow-x: auto;
    overflow-y: hidden;
  }

  .sidebar-resizer {
    flex: none;
    width: 6px;
    margin-left: -3px;
    margin-right: -3px;
    align-self: stretch;
    min-height: 100%;
    position: relative;
    z-index: 6;
    cursor: col-resize;
  }

  .sidebar-resizer::before {
    content: '';
    position: absolute;
    left: 2px;
    top: 0;
    bottom: 0;
    width: 2px;
    border-radius: 999px;
    background: #dbe2ec;
  }

  .sidebar-resizer:hover::before {
    background: #93c5fd;
  }

  .workspace-center {
    padding-left: 0;
    min-height: 0;
    display: flex;
    flex-direction: column;
    height: 100%;
  }

  .workspace-sidebar {
    min-height: 0;
    display: flex;
    flex-direction: column;
    height: 100%;
  }

  .workspace-sidebar .tile,
  .workspace-center .tile {
    height: 100%;
    display: flex;
    flex-direction: column;
  }

  .workspace-sidebar .tile-content,
  .workspace-center .tile-content {
    flex: 1 1 auto;
    min-height: 0;
  }

  .minimap-panel {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    overflow: visible;
  }

  .workspace-sidebar .tile-content {
    overflow: visible;
  }

  .session-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 4px 8px 4px 10px;
    border-radius: 999px;
    background: #f8fafc;
    border: 1px solid #d5dde8;
    color: #475569;
  }

  .session-chip-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #94a3b8;
  }

  .session-chip-value {
    font-size: 12px;
    font-weight: 700;
    color: #1e293b;
  }

  .session-chip-close {
    width: 22px;
    height: 22px;
    border: 0;
    border-radius: 999px;
    background: #e2e8f0;
    color: #475569;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }

  .saved-axes-toggle {
    margin-left: 4px;
    width: 28px;
    height: 28px;
    color: #0f172a;
    border-color: #d5dde8;
  }

  .saved-axes-scrim {
    position: fixed;
    inset: 0;
    background: transparent;
    border: 0;
    margin: 0;
    padding: 0;
    z-index: 120;
    cursor: pointer;
  }

  .saved-axes-drawer {
    position: fixed;
    top: 0;
    right: 0;
    bottom: 0;
    width: min(340px, 92vw);
    background: #ffffff;
    border-left: 1px solid #dbe2ec;
    box-shadow: -14px 0 28px rgba(15, 23, 42, 0.16);
    z-index: 121;
    display: flex;
    flex-direction: column;
    transform: translateX(100%);
    transition: transform 180ms ease;
  }

  .saved-axes-drawer.open {
    transform: translateX(0);
  }

  .saved-axes-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 12px 12px 8px;
    border-bottom: 1px solid #e2e8f0;
  }

  .saved-axes-title {
    font-size: var(--font-size-title);
    font-weight: 700;
    color: #1e293b;
  }

  .saved-axes-close {
    width: 26px;
    height: 26px;
    border: 1px solid #d5dde8;
    border-radius: 6px;
    background: #ffffff;
    color: #475569;
    font-size: 16px;
    line-height: 1;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }

  .saved-axes-error {
    margin: 10px 12px 0;
    color: #dc2626;
    font-size: var(--font-size-small);
    line-height: 1.3;
  }

  .saved-library-sections {
    flex: 1 1 auto;
    min-height: 0;
    display: grid;
    grid-template-rows: repeat(3, minmax(0, 1fr));
  }

  .saved-library-section {
    min-height: 0;
    display: flex;
    flex-direction: column;
    border-top: 1px solid #eef2f7;
  }

  .saved-library-section:first-child {
    border-top: 0;
  }

  .saved-library-section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 10px 12px 4px;
  }

  .saved-library-section-title {
    font-size: var(--font-size-body);
    font-weight: 700;
    color: #1e293b;
  }

  .saved-library-section-meta {
    font-size: var(--font-size-small);
    color: #64748b;
  }

  .saved-axes-body {
    flex: 1 1 auto;
    overflow: auto;
    padding: 10px 12px;
    display: grid;
    align-content: start;
    gap: 8px;
  }

  .saved-axis-row {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 6px;
    align-items: start;
  }

  .saved-axis-load {
    text-align: left;
    border: 1px solid #dbe2ec;
    border-radius: 8px;
    background: #ffffff;
    padding: 8px 9px;
    color: #0f172a;
    min-width: 0;
  }

  .saved-axis-load:disabled {
    opacity: 0.65;
  }

  .saved-axis-name {
    font-size: var(--font-size-body);
    font-weight: 600;
    color: #0f172a;
    line-height: 1.15;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .saved-axis-origin {
    margin-top: 2px;
    font-size: var(--font-size-small);
    color: #64748b;
    line-height: 1.15;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .saved-axis-delete {
    width: 26px;
    height: 26px;
    border: 1px solid #efc8d0;
    border-radius: 6px;
    background: #fff7f8;
    color: #d46d7f;
    font-size: 15px;
    line-height: 1;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }

  .saved-axes-empty {
    border: 1px dashed #dbe2ec;
    border-radius: 8px;
    padding: 12px 10px;
    color: #64748b;
    font-size: var(--font-size-small);
    text-align: center;
  }

  @media (max-width: 1080px) {
    .workspace-grid {
      flex-direction: column;
      overflow-x: visible;
      gap: 12px;
    }

    .workspace-sidebar {
      width: 100% !important;
      min-width: 0 !important;
    }

    .sidebar-resizer {
      display: none;
    }

    .workspace-center {
      padding-left: 0;
      height: auto;
    }

    .workspace-sidebar,
    .workspace-center {
      height: auto;
    }

    .workspace-sidebar .tile,
    .workspace-center .tile {
      height: auto;
    }
  }
</style>
