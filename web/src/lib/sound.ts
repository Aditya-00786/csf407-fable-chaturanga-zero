/** Tiny synthesised sounds (no audio files): a wooden clack for moves, a brass ting for check. */

import { useSettings } from '../state/settings'

let ctx: AudioContext | null = null

function audio(): AudioContext | null {
  if (!useSettings.getState().sound) return null
  try {
    ctx ??= new AudioContext()
    if (ctx.state === 'suspended') void ctx.resume()
    return ctx
  } catch {
    return null
  }
}

function clack(ac: AudioContext, when: number, gain: number, pitch: number) {
  // A short burst of band-passed noise sounds like wood on wood.
  const length = Math.floor(ac.sampleRate * 0.05)
  const buffer = ac.createBuffer(1, length, ac.sampleRate)
  const data = buffer.getChannelData(0)
  for (let i = 0; i < length; i++) data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / length, 4)
  const src = ac.createBufferSource()
  src.buffer = buffer
  const band = ac.createBiquadFilter()
  band.type = 'bandpass'
  band.frequency.value = pitch
  band.Q.value = 3
  const amp = ac.createGain()
  amp.gain.value = gain
  src.connect(band).connect(amp).connect(ac.destination)
  src.start(when)
}

export function playMoveSound(kind: 'move' | 'capture' | 'check' | 'end') {
  const ac = audio()
  if (!ac) return
  const t = ac.currentTime
  if (kind === 'capture') {
    clack(ac, t, 0.9, 900)
    clack(ac, t + 0.045, 0.6, 1400)
    return
  }
  clack(ac, t, 0.7, 1150)
  if (kind === 'check' || kind === 'end') {
    const osc = ac.createOscillator()
    const amp = ac.createGain()
    osc.type = 'sine'
    osc.frequency.value = kind === 'end' ? 523.25 : 880
    amp.gain.setValueAtTime(0.0001, t + 0.05)
    amp.gain.exponentialRampToValueAtTime(0.12, t + 0.07)
    amp.gain.exponentialRampToValueAtTime(0.0001, t + (kind === 'end' ? 1.4 : 0.7))
    osc.connect(amp).connect(ac.destination)
    osc.start(t + 0.05)
    osc.stop(t + 1.5)
  }
}
