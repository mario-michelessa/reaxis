/*
AnimatedMinimap (React)

Drop-in minimap that animates points between positions.

Props:
- items: Array<{ id, url, x, y, gx, gy }>
- prevItems?: Array<{ id, url, x, y, gx, gy }> // previous embedding state
- width?: number (default 200)
- height?: number (default 200)
- gridSize?: number (server n_layer; affects marker size)
- viewFrac?: number (default 0.35)
- minImagePx?: number (default 10)
- mode?: 'original' | 'grid' (target positions in current items)
- fromMode?: 'original' | 'grid' (source positions in prevItems)
- duration?: number (ms; default 700)
- easing?: 'linear' | 'easeInOutCubic' (default 'easeInOutCubic')

Usage note: This repo uses Svelte; this component is provided for React apps
or for future integration via a wrapper/custom element. It has no external
dependencies.
*/
import React, { useEffect, useMemo, useRef, useState } from 'react';

function easeInOutCubic(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function posFromItem(it, which) {
  if (!it) return { x: 0, y: 0 };
  if (which === 'grid') return { x: Number(it.gx ?? it.x ?? 0), y: Number(it.gy ?? it.y ?? 0) };
  return { x: Number(it.x ?? 0), y: Number(it.y ?? 0) };
}

export default function AnimatedMinimap({
  items,
  prevItems,
  width = 200,
  height = 200,
  gridSize = 0,
  viewFrac = 0.35,
  minImagePx = 10,
  mode = 'grid',
  fromMode = 'grid',
  duration = 700,
  easing = 'easeInOutCubic',
}) {
  const [hovering, setHovering] = useState(false);
  const [cx, setCx] = useState(0.5);
  const [cy, setCy] = useState(0.5);
  const [vf, setVf] = useState(viewFrac);
  const animRef = useRef(null);
  const startRef = useRef(0);
  const [frame, setFrame] = useState(0);

  const prevMap = useMemo(() => {
    const m = new Map();
    (prevItems || []).forEach((it) => m.set(it.id, it));
    return m;
  }, [prevItems]);

  // Build animation tracks
  const tracks = useMemo(() => {
    const m = new Map();
    (items || []).forEach((it) => {
      const prev = prevMap.get(it.id) || null;
      const from = prev ? posFromItem(prev, fromMode) : posFromItem(it, mode);
      const to = posFromItem(it, mode);
      m.set(it.id, { id: it.id, url: it.url, from, to });
    });
    return m;
  }, [items, prevMap, mode, fromMode]);

  // Restart animation whenever tracks change
  useEffect(() => {
    cancelAnimationFrame(animRef.current);
    startRef.current = performance.now();
    const tick = () => {
      setFrame((f) => f + 1);
      animRef.current = requestAnimationFrame(tick);
    };
    animRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animRef.current);
  }, [tracks]);

  const tNorm = (() => {
    const elapsed = Math.max(0, performance.now() - startRef.current);
    const raw = Math.min(1, duration > 0 ? elapsed / duration : 1);
    if (easing === 'linear') return raw;
    return easeInOutCubic(raw);
  })();

  // Derived layout
  const cellPx = gridSize && gridSize > 0 ? Math.min(width, height) / gridSize : 16;
  const imSize = Math.max(1, Math.floor(cellPx * 0.95));
  const showDot = imSize < minImagePx;

  // View window
  // keep local zoom synchronized with prop changes
  useEffect(() => {
    setVf(viewFrac);
  }, [viewFrac]);

  const viewW = vf;
  const viewH = vf;
  const x0 = Math.max(0, Math.min(1 - viewW, cx - viewW / 2));
  const y0 = Math.max(0, Math.min(1 - viewH, cy - viewH / 2));
  const x1 = x0 + viewW;
  const y1 = y0 + viewH;

  function onMove(e) {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    setCx(Math.min(1, Math.max(0, x)));
    setCy(Math.min(1, Math.max(0, y)));
  }

  // Build render items at eased t
  const renderItems = useMemo(() => {
    const out = [];
    tracks.forEach((tr) => {
      const x = lerp(tr.from.x, tr.to.x, tNorm);
      const y = lerp(tr.from.y, tr.to.y, tNorm);
      out.push({ id: tr.id, url: tr.url, x, y });
    });
    return out;
  }, [tracks, tNorm, frame]);

  const windowItems = useMemo(() => {
    return renderItems
      .filter((it) => it.x >= x0 && it.x <= x1 && it.y >= y0 && it.y <= y1)
      .map((it) => ({ ...it, lx: (it.x - x0) / viewW, ly: (it.y - y0) / viewH }));
  }, [renderItems, x0, x1, y0, y1, viewW, viewH]);

  return (
    <div className="animated-minimap">
      <div
        role="img"
        aria-label="Animated image scatter minimap"
        className="relative border border-gray-300 bg-white select-none"
        style={{ width: `${width}px`, height: `${height}px`, position: 'relative' }}
        onMouseMove={onMove}
        onMouseEnter={() => setHovering(true)}
        onMouseLeave={() => setHovering(false)}
      >
        {renderItems.map((it) => (
          showDot ? (
            <div
              key={it.id}
              className="absolute rounded-full bg-gray-700 border border-white/70"
              style={{
                left: `${it.x * 100}%`,
                top: `${it.y * 100}%`,
                transform: 'translate(-50%, -50%)',
                width: '6px',
                height: '6px',
              }}
              aria-hidden="true"
            />
          ) : (
            <img
              key={it.id}
              alt=""
              src={it.url}
              className="absolute object-cover rounded"
              draggable={false}
              style={{
                left: `${it.x * 100}%`,
                top: `${it.y * 100}%`,
                transform: 'translate(-50%, -50%)',
                width: `${imSize}px`,
                height: `${imSize}px`,
                pointerEvents: 'none',
              }}
            />
          )
        ))}

        {/* Viewport rectangle */}
        <div
          className="absolute border border-blue-500/70 pointer-events-none"
          style={{
            left: `${x0 * 100}%`,
            top: `${y0 * 100}%`,
            width: `${viewW * 100}%`,
            height: `${viewH * 100}%`,
          }}
        />

        {/* +/- zoom controls */}
        <div className="absolute top-1 right-1 flex gap-1 bg-white text-white rounded p-0 backdrop-blur-sm" style={{ display: 'flex' }}>
          <button type="button" className="w-6 h-6 rounded grid place-items-center m-0 bg-black/20 hover:bg-black/30" onClick={(e) => { e.stopPropagation(); setVf((x)=>Math.max(0.08, Math.min(0.8, x * 0.85))); }} aria-label="Zoom in">+</button>
          <button type="button" className="w-6 h-6 rounded grid place-items-center m-0 bg-black/20 hover:bg-black/30" onClick={(e) => { e.stopPropagation(); setVf((x)=>Math.max(0.08, Math.min(0.8, x / 0.85))); }} aria-label="Zoom out">−</button>
        </div>
      </div>

      {/* Zoom overlay */}
      {hovering && (
        <div className="fixed inset-0 pointer-events-none" style={{ position: 'fixed', inset: 0, pointerEvents: 'none' }}>
          <div className="grid place-content-center z-50" style={{ display: 'grid', placeContent: 'center' }}>
            <div className="relative pointer-events-auto rounded shadow-lg border border-gray-200 bg-white p-2">
              <div className="text-sm mb-2 text-gray-700">Zoom</div>
              <div className="relative bg-white" style={{ width: '600px', height: '600px', position: 'relative' }}>
                {windowItems.map((it) => (
                  <img
                    key={it.id}
                    alt=""
                    src={it.url}
                    className="absolute object-cover rounded"
                    style={{
                      left: `${it.lx * 100}%`,
                      top: `${it.ly * 100}%`,
                      transform: 'translate(-50%, -50%)',
                      width: `${Math.max(12, Math.floor((cellPx * 600 / Math.max(1, width)) * (0.98 / viewFrac)))}px`,
                      height: `${Math.max(12, Math.floor((cellPx * 600 / Math.max(1, width)) * (0.98 / viewFrac)))}px`,
                    }}
                    draggable={false}
                  />
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
