import { writable } from 'svelte/store'

function createAxisBuildersStore() {
  const { subscribe, set, update } = writable([])

  return {
    subscribe,
    reset() {
      set([])
    },
    replace(next) {
      set(Array.isArray(next) ? next : [])
    },
    upsert(session) {
      if (!session || !session.axisId) return
      update((list) => {
        let found = false
        const next = (Array.isArray(list) ? list : []).map((entry) => {
          if (entry?.axisId !== session.axisId) return entry
          found = true
          return session
        })
        return found ? next : [session, ...next]
      })
    },
    patch(axisId, updater) {
      if (!axisId || typeof updater !== 'function') return
      update((list) => (Array.isArray(list) ? list : []).map((entry) => {
        if (entry?.axisId !== axisId) return entry
        const next = updater(entry)
        return next || entry
      }))
    },
    remove(axisId) {
      if (!axisId) return
      update((list) => (Array.isArray(list) ? list : []).filter((entry) => entry?.axisId !== axisId))
    },
  }
}

export const axisBuildersStore = createAxisBuildersStore()
