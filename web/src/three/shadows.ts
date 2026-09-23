/**
 * Shadow maps are redrawn only when something that casts a shadow changes (a piece moves, lifts,
 * morphs, appears or disappears). Camera-only motion such as a board flip then costs one scene
 * render per frame instead of two.
 */

let dirty = true

export function markShadowsDirty() {
  dirty = true
}

export function takeShadowsDirty() {
  const was = dirty
  dirty = false
  return was
}
