export type ImageItem = {
  id: string
  url: string
  className: string
  label: string
  embed: [number, number] // pseudo-embedding for demo
}

// Deterministic pseudo-random from string
function hash2D(input: string): [number, number] {
  let h1 = 2166136261
  let h2 = 2166136261
  for (let i = 0; i < input.length; i++) {
    h1 ^= input.charCodeAt(i)
    h1 += (h1 << 1) + (h1 << 4) + (h1 << 7) + (h1 << 8) + (h1 << 24)
    h2 ^= (input.charCodeAt(i) + 0x9e3779b9) >>> 0
    h2 += (h2 << 1) + (h2 << 4) + (h2 << 7) + (h2 << 8) + (h2 << 24)
  }
  // map to [0,1]
  const x = ((h1 >>> 0) % 10000) / 10000
  const y = ((h2 >>> 0) % 10000) / 10000
  return [x, y]
}

export function generateDemoImages(count = 48): ImageItem[] {
  const classes = [
    'Black_Footed_Albatross',
    'Blue_Jay',
    'Cardinal',
    'Goldfinch',
  ]
  const items: ImageItem[] = []
  for (let i = 0; i < count; i++) {
    const id = `img_${i}`
    const className = classes[i % classes.length]
    const [x, y] = hash2D(id)
    const url = `https://picsum.photos/seed/${encodeURIComponent(id)}/480/320`
    items.push({ id, url, className, label: `${className} #${i + 1}`, embed: [x, y] })
  }
  return items
}

export function l2(a: [number, number], b: [number, number]) {
  const dx = a[0] - b[0]
  const dy = a[1] - b[1]
  return Math.hypot(dx, dy)
}

export function sortBySimilarity(items: ImageItem[], refId: string): ImageItem[] {
  const ref = items.find((i) => i.id === refId)
  if (!ref) return items
  return [...items].sort((a, b) => l2(a.embed, ref.embed) - l2(b.embed, ref.embed))
}

export function clusterKMeans(items: ImageItem[], k = 3): { label: string; items: ImageItem[] }[] {
  if (items.length === 0) return []
  // initialize centroids by first k items
  let centroids = items.slice(0, k).map((i) => [...i.embed] as [number, number])
  let assignments = new Array(items.length).fill(0)
  for (let iter = 0; iter < 8; iter++) {
    // assign
    assignments = items.map((it) => {
      let best = 0
      let bestD = Infinity
      for (let c = 0; c < k; c++) {
        const d = l2(it.embed, centroids[c])
        if (d < bestD) {
          bestD = d
          best = c
        }
      }
      return best
    })
    // recompute
    const sums = Array.from({ length: k }, () => [0, 0, 0]) // [x, y, n]
    assignments.forEach((c, idx) => {
      sums[c][0] += items[idx].embed[0]
      sums[c][1] += items[idx].embed[1]
      sums[c][2] += 1
    })
    centroids = sums.map((s, idx) => (s[2] ? [s[0] / s[2], s[1] / s[2]] : centroids[idx])) as [number, number][]
  }
  // group
  const groups: { [key: number]: ImageItem[] } = {}
  assignments.forEach((c, idx) => {
    groups[c] = groups[c] || []
    groups[c].push(items[idx])
  })
  return Object.entries(groups).map(([c, arr]) => ({ label: `Cluster ${Number(c) + 1}`, items: arr }))
}

