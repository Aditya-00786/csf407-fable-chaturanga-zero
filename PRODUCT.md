# Product

## Register

product

## Users
Two groups, in this order:
1. **Evaluators and audiences at project demos** (CS F407 checkpoint and final presentation). They watch on a
   projector or a laptop while a team member drives. They have a few minutes to understand
   what the game is, see that the engine is correct, and (later) see the AI play.
2. **Newcomers who have never heard of Chaturanga.** They are curious, at a laptop, and want to learn
   the pieces and rules and then play a game, against a friend or the computer.

The team also uses the app day to day to test the engine and, after Milestone 3, to watch
self-play training live. That is a secondary use and must not clutter the primary experience.

## Product Purpose
A browser app for ChaturangZero, the Team Fable AlphaZero-style agent. It lets people:
- learn Chaturanga through interactive lessons;
- play it (Human vs Human, Human vs Computer, Computer vs Computer);
- later, watch the agent train and think.

Success means a first-time visitor can go from "what is this?" to "I finished a game and
understood why it ended" without outside help, and a demo audience remembers the game.
All rules come from the project's Python engine (docs/RULES.md), so the UI never contradicts it.

## Brand Personality
**Scholarly, crafted, calm.** It should feel like a well-made museum exhibit you can play with.
- **Voice:** quietly confident and precise. It explains history and rules with sources, without lecturing.
- **Materials over decoration:** carved wood, brass, ivory and lamp light, not ornament stuck on top.
- **Small, authentic touches of India** are welcome:
  - the traditional Ashtapada board markings;
  - the Sanskrit piece names in real Devanagari;
  - restrained motifs drawn from the pieces themselves.

## Anti-references
- **Generic chess sites** (chess.com / lichess look-alikes): green-and-white checkered boards,
  Staunton pieces, engine-analysis clutter. Chaturanga's board is uncheckered and its pieces are
  elephants, chariots and horses.
- **Tourist-kitsch "exotic India":** mandalas and paisley everywhere, over-saturated saffron,
  fake "Sanskrit-style" Latin fonts.
- **Neon gamer / esports UI:** glowing cyber accents, RGB, aggressive HUDs.
- **Flat SaaS dashboard:** cards on grey, admin-panel feel.

## Design Principles
1. **The board is the hero.** The 3D board and pieces carry the atmosphere. UI chrome stays
   quiet and gets out of the way.
2. **Show, don't tell.** Every rule is taught by moving a piece on a real board, not by a
   paragraph alone.
3. **Never contradict the engine.** Legal moves, results and explanations come from the rules
   engine. The UI only presents them.
4. **Explain every ending.** When a game ends, say exactly why, in plain words, with the rule
   behind it. The stalemate and bare-king rules surprise chess players.
5. **Demo-ready at any moment.** Readable on a projector, fast to reset, and no dead ends or
   half-built screens in the main flows.

## Accessibility & Inclusion
- **Contrast:** WCAG 2.2 AA for all UI text and controls, readable on a projector.
- **Keyboard:** the whole game can be played by keyboard (square cursor plus Enter).
- **Colour-blind safety:** move hints, check and last-move highlights never rely on colour
  alone; they also differ in shape or pattern.
- **Reduced motion:** `prefers-reduced-motion` switches piece travel to an instant fade and
  turns off the camera fly-throughs.
- **Low-end machines:** a flat top-down 2D view for slow laptops and for slides.
