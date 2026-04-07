function normalizeText(value) {
  return String(value || '').trim()
}

function normalizeStringArray(values) {
  if (!Array.isArray(values)) return []
  return values.map((value) => normalizeText(value)).filter(Boolean)
}

function normalizeNumberArray(values) {
  if (!Array.isArray(values)) return []
  return values.map((value) => Number(value) || 0)
}

function cloneAxis(axis, meta) {
  const source = axis && typeof axis === 'object' ? axis : {}
  return {
    ...source,
    name: meta.axisName,
    q: meta.query,
    mode: meta.mode,
    model_type: meta.modelType,
    modelType: meta.modelType,
    semantic_method: meta.semanticMethod,
    semanticMethod: meta.semanticMethod,
    norm: meta.norm,
    semantic_embedding_file: meta.semanticEmbeddingFile,
    semanticEmbeddingFile: meta.semanticEmbeddingFile,
    debug: { ...(meta.debugInfo || {}) },
    coords: source.coords && typeof source.coords === 'object' ? { ...source.coords } : {},
    labels: Array.isArray(source.labels) ? [...source.labels] : [],
    label_positions: Array.isArray(source.label_positions) ? [...source.label_positions] : [],
    w0Summary: meta.w0Summary,
    pos_prompt_ensemble: meta.posPromptEnsemble,
    neg_prompt_ensemble: meta.negPromptEnsemble,
    prompt_ensemble: meta.promptEnsemble,
  }
}

export function axisSessionFromResponse(current, data, options = {}) {
  const axis = data?.axis
  if (!axis?.id || !axis?.coords) return null

  const currentSession = current && typeof current === 'object' ? current : {}
  const debugInfo = (data?.debug && typeof data.debug === 'object') ? data.debug : {}
  const rawSummary = (data?.w0_summary && typeof data.w0_summary === 'object') ? data.w0_summary : {}
  const posPromptEnsemble = normalizeStringArray(rawSummary.pos_prompts)
  const negPromptEnsemble = normalizeStringArray(rawSummary.neg_prompts)
  const promptEnsemble = normalizeStringArray(rawSummary.prompts)
  const preferredName = normalizeText(
    options.preferredName
    || currentSession?.axis?.name
    || currentSession?.q
    || data?.q
    || axis?.name,
  )
  const query = normalizeText(currentSession?.q || data?.q || preferredName || axis?.name)
  const axisName = preferredName || normalizeText(axis?.name) || query || normalizeText(axis?.id)
  const mode = normalizeText(data?.mode || currentSession?.mode)
  const modelType = normalizeText(data?.model_type || currentSession?.modelType)
  const semanticMethod = normalizeText(data?.semantic_method || axis?.semantic_method || currentSession?.semanticMethod)
  const norm = Boolean('norm' in (data || {}) ? data.norm : currentSession?.norm)
  const semanticEmbeddingFile = normalizeText(
    data?.semantic_embedding_file
    || debugInfo?.semantic_embedding_file
    || currentSession?.semanticEmbeddingFile,
  )
  const w0Summary = {
    ...(rawSummary || {}),
    pos_prompts: posPromptEnsemble,
    neg_prompts: negPromptEnsemble,
    prompts: promptEnsemble,
    mode,
    model_type: modelType,
    semantic_method: semanticMethod,
    norm,
    semantic_embedding_file: semanticEmbeddingFile,
  }

  return {
    axisId: normalizeText(axis.id),
    q: query,
    mode,
    modelType,
    semanticMethod,
    norm,
    semanticEmbeddingFile,
    debugInfo: { ...debugInfo },
    axis: cloneAxis(axis, {
      axisName,
      query,
      mode,
      modelType,
      semanticMethod,
      norm,
      semanticEmbeddingFile,
      debugInfo,
      w0Summary,
      posPromptEnsemble,
      negPromptEnsemble,
      promptEnsemble,
    }),
    ids: normalizeStringArray(data?.ids),
    projectionValues: normalizeNumberArray(data?.projection_values),
    projectionMin: Number(data?.projection_min || 0),
    projectionMax: Number(data?.projection_max || 0),
    scores: normalizeNumberArray(data?.scores),
    std: normalizeNumberArray(data?.std),
    decileExemplars: Array.isArray(data?.decile_exemplars) ? data.decile_exemplars : [],
    hotspots: Array.isArray(data?.hotspots) ? data.hotspots : [],
    moveCount: Number(data?.move_count || 0),
    maxMoves: Number(data?.max_moves || 0),
    moves: Array.isArray(data?.moves) ? data.moves : [],
    undefinedIds: normalizeStringArray(data?.undefined_ids),
    w0Summary,
    posPromptEnsemble,
    negPromptEnsemble,
    promptEnsemble,
    moveHistory: Array.isArray(currentSession?.moveHistory) ? currentSession.moveHistory : [],
  }
}

export function axisPayloadFromSession(session) {
  if (!session?.axis?.id) return null
  const axis = session.axis
  return {
    ...axis,
    coords: axis.coords && typeof axis.coords === 'object' ? { ...axis.coords } : {},
    labels: Array.isArray(axis.labels) ? [...axis.labels] : [],
    label_positions: Array.isArray(axis.label_positions) ? [...axis.label_positions] : [],
    q: normalizeText(session.q || axis.q || axis.name || axis.id),
    mode: normalizeText(session.mode || axis.mode),
    model_type: normalizeText(session.modelType || axis.model_type || axis.modelType),
    modelType: normalizeText(session.modelType || axis.model_type || axis.modelType),
    semantic_method: normalizeText(session.semanticMethod || axis.semantic_method || axis.semanticMethod),
    semanticMethod: normalizeText(session.semanticMethod || axis.semantic_method || axis.semanticMethod),
    norm: Boolean(typeof session.norm === 'boolean' ? session.norm : axis.norm),
    semantic_embedding_file: normalizeText(session.semanticEmbeddingFile || axis.semantic_embedding_file || axis.semanticEmbeddingFile),
    semanticEmbeddingFile: normalizeText(session.semanticEmbeddingFile || axis.semantic_embedding_file || axis.semanticEmbeddingFile),
    debug: session.debugInfo && typeof session.debugInfo === 'object' ? { ...session.debugInfo } : {},
    w0Summary: session.w0Summary && typeof session.w0Summary === 'object' ? { ...session.w0Summary } : {},
    pos_prompt_ensemble: normalizeStringArray(session.posPromptEnsemble || axis.pos_prompt_ensemble || axis.posPromptEnsemble),
    neg_prompt_ensemble: normalizeStringArray(session.negPromptEnsemble || axis.neg_prompt_ensemble || axis.negPromptEnsemble),
    prompt_ensemble: normalizeStringArray(session.promptEnsemble || axis.prompt_ensemble || axis.promptEnsemble),
  }
}
