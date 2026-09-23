/** Per-viewer display preferences, remembered in localStorage when available. */

import { flushSync } from 'react-dom'
import { create } from 'zustand'

export type Theme = 'dark' | 'light'
export type LabelMode = 'off' | 'english' | 'sanskrit'
export type ViewMode = '3d' | 'flat'
/** How much supporting text to show around the board: headlines only, or hints, counters and rule notes too. */
export type DetailLevel = 'essential' | 'full'
/** Individually switchable parts of the Full detail level. */
export type DetailPart = 'hints' | 'counters' | 'explanation' | 'rules'
export type DetailParts = Record<DetailPart, boolean>

const ALL_PARTS: DetailParts = { hints: true, counters: true, explanation: true, rules: true }

interface Settings {
  theme: Theme
  sound: boolean
  labels: LabelMode
  view: ViewMode
  details: DetailLevel
  detailParts: DetailParts
  setTheme: (t: Theme) => void
  setSound: (on: boolean) => void
  setLabels: (l: LabelMode) => void
  setView: (v: ViewMode) => void
  setDetails: (d: DetailLevel) => void
  setDetailPart: (part: DetailPart, on: boolean) => void
}

function read(key: string): string | null {
  try {
    return localStorage.getItem(key)
  } catch {
    return null
  }
}

function write(key: string, value: string) {
  try {
    localStorage.setItem(key, value)
  } catch {
    /* private mode or blocked storage: preferences just aren't remembered */
  }
}

function readParts(): DetailParts {
  try {
    const saved = JSON.parse(read('cz-detail-parts') ?? '{}')
    return { ...ALL_PARTS, ...(typeof saved === 'object' && saved ? saved : {}) }
  } catch {
    return { ...ALL_PARTS }
  }
}

const initialTheme = (document.documentElement.dataset.theme as Theme) === 'light' ? 'light' : 'dark'

export const useSettings = create<Settings>((set, get) => ({
  theme: initialTheme,
  sound: read('cz-sound') !== 'off',
  labels: (read('cz-labels') as LabelMode) ?? 'off',
  view: (read('cz-view') as ViewMode) === 'flat' ? 'flat' : '3d',
  details: read('cz-details') === 'full' ? 'full' : 'essential',
  detailParts: readParts(),
  setTheme: (theme) => {
    document.documentElement.dataset.theme = theme
    write('cz-theme', theme)
    set({ theme })
  },
  setSound: (sound) => {
    write('cz-sound', sound ? 'on' : 'off')
    set({ sound })
  },
  setLabels: (labels) => {
    write('cz-labels', labels)
    set({ labels })
  },
  setView: (view) => {
    write('cz-view', view)
    set({ view })
  },
  setDetails: (details) => {
    write('cz-details', details)
    set({ details })
  },
  setDetailPart: (part, on) => {
    const detailParts = { ...get().detailParts, [part]: on }
    write('cz-detail-parts', JSON.stringify(detailParts))
    set({ detailParts })
  },
}))

/** Whether a detail part is visible: only at the Full level, and only if its own toggle is on. */
export function useDetail(part: DetailPart) {
  return useSettings((s) => s.details === 'full' && s.detailParts[part])
}

export function prefersReducedMotion() {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

/**
 * Switch theme with a compositor crossfade (View Transitions API) where available. The page is
 * snapshotted once and faded on the GPU, which stays smooth on slow machines; the 3D scene eases
 * its own lighting underneath. Without support, or with reduced motion, it switches instantly.
 */
export function switchTheme(next: Theme) {
  const apply = () => flushSync(() => useSettings.getState().setTheme(next))
  const doc = document as Document & { startViewTransition?: (cb: () => void) => unknown }
  if (!doc.startViewTransition || prefersReducedMotion()) {
    apply()
    return
  }
  doc.startViewTransition(apply)
}
