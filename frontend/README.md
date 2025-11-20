# Svelte + Vite

This template should help get you started developing with Svelte in Vite.

## Recommended IDE Setup

[VS Code](https://code.visualstudio.com/) + [Svelte](https://marketplace.visualstudio.com/items?itemName=svelte.svelte-vscode).

## Need an official Svelte framework?

Check out [SvelteKit](https://github.com/sveltejs/kit#readme), which is also powered by Vite. Deploy anywhere with its serverless-first approach and adapt to various platforms, with out of the box support for TypeScript, SCSS, and Less, and easily-added support for mdsvex, GraphQL, PostCSS, Tailwind CSS, and more.

## Technical considerations

**Why use this over SvelteKit?**

- It brings its own routing solution which might not be preferable for some users.
- It is first and foremost a framework that just happens to use Vite under the hood, not a Vite app.

This template contains as little as possible to get started with Vite + Svelte, while taking into account the developer experience with regards to HMR and intellisense. It demonstrates capabilities on par with the other `create-vite` templates and is a good starting point for beginners dipping their toes into a Vite + Svelte project.

Should you later need the extended capabilities and extensibility provided by SvelteKit, the template has been structured similarly to SvelteKit so that it is easy to migrate.

**Why `global.d.ts` instead of `compilerOptions.types` inside `jsconfig.json` or `tsconfig.json`?**

Setting `compilerOptions.types` shuts out all other types not explicitly listed in the configuration. Using triple-slash references keeps the default TypeScript setting of accepting type information from the entire workspace, while also adding `svelte` and `vite/client` type information.

**Why include `.vscode/extensions.json`?**

Other templates indirectly recommend extensions via the README, but this file allows VS Code to prompt the user to install the recommended extension upon opening the project.

**Why enable `checkJs` in the JS template?**

It is likely that most cases of changing variable types in runtime are likely to be accidental, rather than deliberate. This provides advanced typechecking out of the box. Should you like to take advantage of the dynamically-typed nature of JavaScript, it is trivial to change the configuration.

**Why is HMR not preserving my local component state?**

HMR state preservation comes with a number of gotchas! It has been disabled by default in both `svelte-hmr` and `@sveltejs/vite-plugin-svelte` due to its often surprising behavior. You can read the details [here](https://github.com/sveltejs/svelte-hmr/tree/master/packages/svelte-hmr#preservation-of-local-state).

If you have state that's important to retain within a component, consider creating an external store which would not be replaced by HMR.

```js
// store.js
// An extremely simple external store
import { writable } from 'svelte/store'
export default writable(0)
```

## React component: AnimatedMinimap

A React implementation of the minimap with animated transitions is available at `frontend/src/react/AnimatedMinimap.jsx`.

Props:
- `items`: array of `{ id, url, x, y, gx, gy }` for the current embedding/mode
- `prevItems` (optional): previous array to animate from when switching embedding types
- `mode`: `'original' | 'grid'` target position type for current items (default `'grid'`)
- `fromMode`: `'original' | 'grid'` source position type for previous items (default `'grid'`)
- `width`, `height`, `gridSize`, `viewFrac`, `minImagePx`, `duration`, `easing`

Example (in a React app):

```jsx
import AnimatedMinimap from './src/react/AnimatedMinimap';

export default function Panel({ items, prevItems, nLayer }) {
  return (
    <AnimatedMinimap
      items={items}
      prevItems={prevItems}
      gridSize={nLayer}
      mode="grid" // animate to snapped grid positions
      fromMode="grid" // animate from prior grid positions
      width={220}
      height={220}
      duration={700}
    />
  );
}
```

Note: This Svelte app does not mount React by default; consume the component in a React project or wrap it as a custom element if needed.

## Standalone mode (no backend)

- Toggle via env or query: set `VITE_STANDALONE=1` or open `/?standalone=1`.
- For a single dataset, place `gallery.json` and assets under `frontend/public/` and ensure `url` fields in `gallery.json` are reachable.
- For multiple datasets, place each dataset under `frontend/public/datasets/<name>/` with a `gallery.json` inside. Optionally include images in the same folder and reference them with relative paths in `gallery.json` (e.g., `"url": "images/cat.jpg"`).
- Create a dataset manifest at `frontend/public/datasets/index.json`:
  - Either an array of strings: `["ISIC2017", "Birds"]`
  - Or objects: `[{"label":"ISIC 2017","value":"ISIC2017"}]`
- In standalone mode, the app fetches `./datasets/<name>/gallery.json` and prefixes relative URLs with `./datasets/<name>/`.
 - To enable metadata axes in standalone (for the "Metadata" projection), generate `gallery_metadata.json` next to your dataset's galleries. Use:
   - `python backend/export_metadata.py <DATASET_DIR> frontend/public/datasets/<NAME>`
   - This writes `frontend/public/datasets/<NAME>/gallery_metadata.json` with per-field axes derived from `metadata.csv` in your dataset root.
   - The app auto-loads `gallery_metadata.json` and merges its `metadata_axes` for all embedding methods.
