import '@unocss/reset/tailwind.css'
import 'virtual:uno.css'
import './app.css' // Global styles should load after resets/utilities
import App from './App.svelte'
import AppStandalone from './App_standalone.svelte'

const isStandalone = (
  (import.meta?.env && String(import.meta.env.VITE_STANDALONE) === '1') ||
  (typeof window !== 'undefined' && window.location && window.location.search && window.location.search.includes('standalone=1'))
)

const Root = isStandalone ? AppStandalone : App
const app = new Root({ target: document.getElementById('app') })

export default app
