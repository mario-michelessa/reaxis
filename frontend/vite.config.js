import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import Unocss from 'unocss/vite';

const backendTarget = process.env.VITE_BACKEND_PROXY_TARGET || 'http://127.0.0.1:5001';

export default defineConfig({
  plugins: [svelte(), Unocss()],
  server: {
    port: 5174,
    strictPort: false,
    cors: true,
    proxy: {
      '/api': {
        target: backendTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  }
});

// import { defineConfig } from 'vite'
// import { svelte } from '@sveltejs/vite-plugin-svelte'

// // https://vite.dev/config/
// export default defineConfig({
//   plugins: [svelte()],
// })
