import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import Unocss from 'unocss/vite';

export default defineConfig({
  plugins: [svelte(), Unocss()],
  server: {
    port: 5174,
    strictPort: false,
    cors: true
  }
});

// import { defineConfig } from 'vite'
// import { svelte } from '@sveltejs/vite-plugin-svelte'

// // https://vite.dev/config/
// export default defineConfig({
//   plugins: [svelte()],
// })
