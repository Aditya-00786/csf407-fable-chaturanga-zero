import { TopBar } from './components/TopBar'
import { useRoute } from './lib/route'
import { Showcase } from './three/Showcase'
import { LearnView } from './views/LearnView'
import { PlayView } from './views/PlayView'
import { RulesView } from './views/RulesView'
import { TrainingView } from './views/TrainingView'

export default function App() {
  const route = useRoute()
  // Development aid: http://localhost:5173/?showcase shows every piece model side by side.
  if (location.search.includes('showcase')) return <div className="h-full"><Showcase /></div>
  return (
    <div className="flex h-full flex-col">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-[var(--z-toast)] focus:rounded-md focus:bg-accent focus:px-3 focus:py-2 focus:text-accent-ink">
        Skip to content
      </a>
      <TopBar page={route.page} />
      <main id="main" className="flex min-h-0 flex-1 flex-col overflow-y-auto lg:overflow-hidden">
        {route.page === 'play' && <PlayView />}
        {route.page === 'learn' && <LearnView lessonId={route.param} />}
        {route.page === 'rules' && <RulesView />}
        {route.page === 'training' && <TrainingView />}
      </main>
    </div>
  )
}
