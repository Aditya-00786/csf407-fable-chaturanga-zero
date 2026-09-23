import { useEffect, useState } from 'react'

/** Whether the page is full screen, a toggle, and whether full screen is available at all. */
export function useFullscreen(): [boolean, () => void, boolean] {
  const [on, setOn] = useState(() => !!document.fullscreenElement)
  useEffect(() => {
    const sync = () => setOn(!!document.fullscreenElement)
    document.addEventListener('fullscreenchange', sync)
    return () => document.removeEventListener('fullscreenchange', sync)
  }, [])
  const toggle = () => {
    if (document.fullscreenElement) void document.exitFullscreen()
    else void document.documentElement.requestFullscreen?.().catch(() => {})
  }
  return [on, toggle, !!document.fullscreenEnabled]
}
