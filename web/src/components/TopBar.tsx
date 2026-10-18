import { Moon, Sun } from 'lucide-react'
import { href, type Page } from '../lib/route'
import { switchTheme, useSettings } from '../state/settings'

const NAV: { page: Page; label: string }[] = [
  { page: 'play', label: 'Play' },
  { page: 'learn', label: 'Learn' },
  { page: 'rules', label: 'Rules' },
  { page: 'training', label: 'Training' },
]

function Mark() {
  // An Ashtapada square with its cross: the traditional board marking.
  return (
    <svg viewBox="0 0 32 32" width="26" height="26" aria-hidden className="text-accent">
      <rect x="3" y="3" width="26" height="26" rx="4" fill="none" stroke="currentColor" strokeOpacity=".55" strokeWidth="1.6" />
      <path d="M9.5 9.5l13 13M22.5 9.5l-13 13" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" />
    </svg>
  )
}

export function TopBar({ page }: { page: Page }) {
  const theme = useSettings((s) => s.theme)
  return (
    <header className="flex h-14 shrink-0 items-center gap-4 border-b border-line bg-surface px-4 sm:gap-6 sm:px-5">
      <a href={href('play')} className="flex items-center gap-2.5 rounded-md" aria-label="ChaturangZero home">
        <Mark />
        <span className="font-serif text-[19px] font-semibold tracking-[-0.01em] text-ink">ChaturangZero</span>
        <span className="hidden font-deva text-[15px] text-ink-faint md:inline" lang="sa">चतुरङ्ग</span>
      </a>
      <nav aria-label="Main" className="flex items-center gap-1">
        {NAV.map((n) => {
          const active = n.page === page
          return (
            <a
              key={n.page}
              href={href(n.page)}
              aria-current={active ? 'page' : undefined}
              className={`relative rounded-md px-2.5 py-1.5 text-sm font-medium transition-colors duration-150 sm:px-3 ${
                active ? 'text-ink' : 'text-ink-muted hover:text-ink'
              }`}
            >
              {n.label}
              {active && <span className="absolute inset-x-2.5 -bottom-[11px] h-0.5 rounded-full bg-accent sm:inset-x-3" />}
            </a>
          )
        })}
      </nav>
      <div className="ml-auto flex items-center gap-2">
        <span className="hidden text-[12.5px] text-ink-faint xl:inline">CS F407 · Team Fable</span>
        <button
          type="button"
          onClick={() => switchTheme(theme === 'dark' ? 'light' : 'dark')}
          className="cz-theme-toggle grid h-9 w-9 place-items-center rounded-lg text-ink-muted transition-colors duration-150 hover:bg-surface-2 hover:text-ink"
          aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
          title={theme === 'dark' ? 'Light theme' : 'Dark theme'}
        >
          {/* Both icons stay mounted; the switch turns one out and the other in (compositor-only properties). */}
          <span className="relative grid h-[18px] w-[18px] place-items-center" aria-hidden>
            <Sun
              size={18}
              className={`absolute transition-[opacity,rotate,scale] duration-[420ms] ease-out-quart ${
                theme === 'dark' ? 'scale-100 rotate-0 opacity-100' : 'scale-50 -rotate-90 opacity-0'
              }`}
            />
            <Moon
              size={18}
              className={`absolute transition-[opacity,rotate,scale] duration-[420ms] ease-out-quart ${
                theme === 'light' ? 'scale-100 rotate-0 opacity-100' : 'scale-50 rotate-90 opacity-0'
              }`}
            />
          </span>
        </button>
      </div>
    </header>
  )
}
