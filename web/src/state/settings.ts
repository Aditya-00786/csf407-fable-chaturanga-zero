/** Per-viewer display preferences, remembered in localStorage when available. */

import { create } from 'zustand'

export type Theme = 'dark' | 'light'
export type LabelMode = 'off' | 'english' | 'sanskrit'
export type ViewMode = '3d' | 'flat'
/** How much supporting text to show around the board: headlines only, or hints, counters and rule notes too. */
export type DetailLevel = 'essential' | 'full'

interface Settings {
  theme: Theme
  sound: boolean
  labels: LabelMode
  view: ViewMode
  details: DetailLevel
  setTheme: (t: Theme) => void
  setSound: (on: boolean) => void
  setLabels: (l: LabelMode) => void
  setView: (v: ViewMode) => void
  setDetails: (d: DetailLevel) => void
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

const initialTheme = (document.documentElement.dataset.theme as Theme) === 'light' ? 'light' : 'dark'

export const useSettings = create<Settings>((set) => ({
  theme: initialTheme,
  sound: read('cz-sound') !== 'off',
  labels: (read('cz-labels') as LabelMode) ?? 'off',
  view: (read('cz-view') as ViewMode) === 'flat' ? 'flat' : '3d',
  details: read('cz-details') === 'full' ? 'full' : 'essential',
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
}))

export function prefersReducedMotion() {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches
}
