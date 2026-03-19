# ReQuest

Interactive UI for creating, refining, and reusing semantic axes over image collections.

This README documents the current live UI. It intentionally excludes hidden or obsolete components.

## Dataset Preparation

Prepared datasets live under `data/datasets/` and are built with:

```bash
python backend/import_curated_dataset.py --preset <preset> --name <output> --limit 1500
python backend/import_local_curated_datasets.py --only <dataset> [<dataset> ...]
```

Current local batch-prepared outputs include:

- `broden1_224`
- `CUB`
- `HAM10000`
- `ImageNet_R`
- `ImageNet_n02958343` (1307 images available in source)
- `Imagenette1500`
- `inat2021birds`
- `MapillaryVistas`

Each prepared dataset is flattened into a single folder and includes:

- `metadata.csv`
- `.cache/embeddings_color_rgb.npz`
- `.cache/embeddings_siglip2.npz`
- `.cache/embeddings_clip.npz`
- `.cache/embeddings_dino.npz`
- `.cache/coords_pca2d_color_rgb.npz`
- `.cache/coords_pca2d_siglip2.npz`
- `.cache/coords_pca2d_clip.npz`
- `.cache/coords_pca2d_dino.npz`

Notes:

- `siglip2` is now the default semantic feature space in the backend and frontend. Older datasets that only have `clip` caches still load through a compatibility fallback until they are backfilled.
- Large source images are resized during import instead of being dropped when they exceed the configured source-pixel ceiling.
- `inat2021birds` carries taxonomy and observation metadata from the source JSON tables.
- `MapillaryVistas` uses the `training` and `validation` image sets plus `v2.0` semantic masks; `testing` is excluded because it has no labels.

## UI Interactions

### Global Workspace

- Change the active dataset from the `Dataset` dropdown in the top bar.
- Open the saved-axis drawer from the hamburger menu in the top-right corner.
- Load a saved axis from the drawer into the current dataset by clicking its row.
- Delete a saved axis from the drawer with the red close button.
- Resize the left sidebar by dragging the vertical divider between the sidebar and the visualization panel.

### Axes Creation Panel

- Enter a free-form visualization request in the prompt box.
- Click the `Analyze` button to run the LLM and extract candidate attributes.
- Hover a suggested attribute chip to highlight the prompt words that supported that attribute.
- Select or deselect suggested attribute chips by clicking them.
- Rename a suggested attribute inline with the edit button on the chip.
- Create axes from the currently selected suggested attributes with the add button at the bottom-right of the suggestion area.
- Add a custom axis directly from the manual text input in the `Axes` section.
- Clear all active histogram slices with the `Unslice` button when slices are active.

### Axis Builder Cards

Each created axis opens an axis-builder card with its own interactions.

- Save the axis to the reusable axis library with the save button.
- Assign the axis to the minimap `X` or `Y` coordinate with the `X` and `Y` buttons.
- Remove the axis entirely with the red close button.
- Inspect the negative anchor prompts from the left anchor button.
- Inspect the positive anchor prompts from the right anchor button.
- Edit either anchor-prompt list from its popup and save the edits to recompute the axis prior.
- Hover a histogram bar to preview representative thumbnails for that value range.
- Drag across the histogram to create a slice on that axis and immediately filter the minimap to that subset.
- Double-click the histogram to clear that axis slice.
- Drag a thumbnail from a rating bin to another bin to refine the axis with user feedback.
- Drag a thumbnail onto the undefined cross to mark that image as undefined for this axis.
- Click the undefined cross to open the undefined-image tooltip.
- Drag an image back out of the undefined tooltip and drop it onto a rating bin to restore it to the axis.
- Click any thumbnail in the builder or in the undefined tooltip to open the large image focus view in the visualization panel.

### Visualization Panel

- Pick the current `X` axis from the dropdown below the canvas.
- Pick the current `Y` axis from the dropdown on the left side of the canvas.
- Resize the minimap by dragging the small resize handle at the bottom-right corner of the panel.
- Scroll on the canvas to zoom the visualization around the cursor.
- Hold `Shift` and scroll to change the size of the local focus rectangle used for gridding.
- Hover the canvas in select mode to preview the local focus rectangle.
- Click the canvas in select mode to open a local gridded view around the hovered area.
- Click outside the active grid to close the gridded view.
- Click an image inside the gridded view to open the large image focus view.

### Visualization Toolbox

- Use `Select` mode to inspect the scatterplot and create the local gridded view.
- Use `Grab` mode to drag an image directly in the visualization and turn that move into axis feedback.
- Use `Lasso` mode to draw a free-form region over the current visible points.
- After drawing a lasso, click `Isolate` to show only the selected images.
- After drawing a lasso, click `Exclude` to hide the selected images.
- Toggle `Density` to overlay a density map on top of the current scatterplot.
- Toggle `Uncertainty` to replace the vertical coordinate with uncertainty values.
- Change `max` to limit how many thumbnails are shown directly in the minimap before the rest are replaced by dots.
- Click `Reset view` to restore the default zoom and close the current local grid.
- Clear an active lasso subset or histogram-based subset from the `[Subset]` chip at the top of the minimap. This restores all images and resets the current view.

### Image Focus View

- Open the focus view by clicking an image inside the minimap grid or by clicking a thumbnail in an axis builder.
- Inspect a larger version of the image in the focus backdrop.
- Adjust any displayed axis slider to move the image along that axis and trigger a backend axis update when the slider is released.
- Close the focus view with the close button or by clicking outside the panel.

## Component Notes

- The left panel is driven by `frontend/src/components/PromptSidebar.svelte`.
- The per-axis refinement cards are implemented in `frontend/src/components/AxisBuilder.svelte`.
- The visualization panel is implemented in `frontend/src/components/AxesMinimap.svelte`.
- The main application shell is implemented in `frontend/src/App.svelte`.
