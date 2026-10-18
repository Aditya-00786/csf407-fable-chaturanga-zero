/** Minimal hash routing: #/play, #/learn/<lesson-id>, #/rules, #/training. */

import { useEffect, useState } from 'react'

export type Page = 'play' | 'learn' | 'rules' | 'training'

export interface Route {
  page: Page
  param: string | null
}

function parse(hash: string): Route {
  const [, page, param] = hash.replace(/^#/, '').split('/')
  if (page === 'learn' || page === 'rules' || page === 'training') return { page, param: param ?? null }
  return { page: 'play', param: null }
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parse(location.hash))
  useEffect(() => {
    const onChange = () => setRoute(parse(location.hash))
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return route
}

export const href = (page: Page, param?: string) => `#/${page}${param ? `/${param}` : ''}`
