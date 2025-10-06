import { defineConfig, presetUno, presetIcons, presetTypography } from 'unocss'

export default defineConfig({
  presets: [
    presetUno(),
    presetIcons(),
    presetTypography(),
  ],
  shortcuts: {
    'btn': 'px-3 py-1 rounded bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed',
    'chip': 'px-2 py-0.5 rounded-full bg-gray-200 dark:bg-gray-700 text-xs',
    'card': 'rounded shadow-sm border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800',
  },
})

