# ChaturangZero — Rules Specification (v1.0 — all rules decided)

This is the rule set the engine implements. Historical sources disagree on several points.
Our method:
1. Every rule cites where it comes from.
2. Where sources clash, we count the references and follow the majority.
3. On a tie, the team decides.

Counting conventions:
- A page that copies another source (same text or same diagram) is counted **once**.
- Sources that describe **shatranj** or **later Indian chess** are listed as *supporting*, not counted.
- A source we could not open ourselves is marked *indirect* and not counted.

Rules marked **[ENG]** are engineering rules with no historical basis. They exist only so that
self-play games always end.

## 1. Sources

Sources with a saved copy in [`docs/Handouts/`](Handouts/) link to that PDF; the rest link to the web.

### Scholarly / primary
| Key | Source |
|-----|--------|
| **[Murray]** | H. J. R. Murray, *A History of Chess*, Oxford: Clarendon Press, 1913 ([archive.org](https://archive.org/details/AHistoryOfChess)). p. 27 n. 4 (king files); pp. 59–61 (early Indian game, al-Adli's Indian rules); pp. 63–65 (translation of **Nilakantha**, a Sanskrit text on two-player chess, c. 1600–1700); p. 72 (four-handed dice game); pp. 80–81 (19th-c. Hindustani chess); pp. 224–229 (shatranj) |
| **[Averbakh]** | Y. Averbakh, *A History of Chess: From Chaturanga to the Present Day*, Russell Enterprises, 2012, pp. 36–37 (shatranj rules) ([PDF](Handouts/a-history-of-chess-from-chaturanga-to-the-present-day_compress.pdf)) |

### Encyclopedic / secondary
| Key | Source | Notes |
|-----|--------|-------|
| **[W-Chat]** | Wikipedia, "Chaturanga" ([PDF](Handouts/Chaturanga%20-%20Wikipedia.pdf); [older snapshot](Handouts/b17318c6171017e0.pdf)) | Cites Murray, Pritchard (2007), ChessZone (dead page) and learn-and-play-online-chess |
| **[W-Promo]** | Wikipedia, "Promotion (chess)" ([link](https://en.wikipedia.org/wiki/Promotion_(chess))) | Mantri-only view cited to chessvariants.com; file-based view cited to Ludoteka |
| **[W-Shat]** | Wikipedia, "Shatranj" ([link](https://en.wikipedia.org/wiki/Shatranj)) | Supporting only |
| **[W-Ind]** | Wikipedia, "Indian chess" ([link](https://en.wikipedia.org/wiki/Indian_chess)) | 18th–19th c. Indian chess; supporting only |
| **[Wikibooks]** | Wikibooks, "Chess Variants/Chaturanga and Shatranj" ([link](https://en.wikibooks.org/wiki/Chess_Variants/Chaturanga_and_Shatranj)) | No citations |
| **[Daou]** | N. D. Daou, *Historical Chess Variants*, 2007, hosted on chessvariants.com ([PDF](Handouts/HistoricalChessVariants.pdf)) | Internet-based compilation. Its footnote 5 ties its promotion rule to 19th-c. Indian chess |

### Websites / game sites
| Key | Source | Notes |
|-----|--------|-------|
| **[Handout]** | Rules handout ([PDF](Handouts/50b2970664273654.pdf)) | Setup diagram is the same as Wikipedia's |
| **[LAPOC]** | learn-and-play-online-chess.com, "Chaturanga – The Original Chess" ([PDF](Handouts/Chaturanga%20-%20The%20Original%20Chess.pdf)) | |
| **[Ludoteka]** | Ludoteka, "Chaturanga – game rules" ([link](http://www.ludoteka.com/chaturanga-en.html)) | Rules match 19th-c. Hindustani chess ([Murray p. 81]) |
| **[ChessCom]** | Chess.com, "Chaturanga" ([link](https://www.chess.com/terms/chaturanga-game)) | |
| **[WorldChess]** | World Chess, "Chaturanga" ([link](https://worldchess.com/chess-terms/chaturanga-game)) | |
| **[ChOnline]** | chaturanga.online, "Rules" ([link](https://chaturanga.online/rules/)) | Mostly about shatranj |
| **[AbsGames]** | *The New Abstract Games*, "Shatranj" ([link](https://www.abstractgames.org/shatranj1.html)) | Shatranj 70-move rule; supporting only |

### Excluded / not counted
| Source | Why |
|---|---|
| Gambiter, "Chaturanga" | Copy of the Wikipedia article |
| chessvariants.com "Chaturanga" page | Could not open (CAPTCHA). Counted as *indirect*: Wikipedia cites it for Mantri-only promotion |
| Front Porch Classics rulebook, gamesrules.net | 4-player dice game (chaturaji), not the 2-player game |
| *Caturangadīpikā* (Śūlapāṇi) | A manual of the 4-player dice game |

## 2. Board and setup

- 8×8 uncheckered board (Ashtapada) [W-Chat], [Murray p. 64].
- **White moves first** [W-Chat], [Ludoteka]. [Handout] lets the players choose; an engine needs a fixed rule.
- **Setup: crosswise — the Rajas do not face each other** (majority, §6.6; team confirmed).
  White's Raja is on e1 with the Mantri on d1. Black's Raja is on d8 with the Mantri on e8.
  In other words, each Mantri stands on its Raja's left.

```
  8 | R N G K M G N R      Black: Raja d8, Mantri e8
  7 | P P P P P P P P
  6 | . . . . . . . .
  5 | . . . . . . . .
  4 | . . . . . . . .
  3 | . . . . . . . .
  2 | P P P P P P P P
  1 | R N G M K G N R      White: Mantri d1, Raja e1
      a b c d e f g h
R = Ratha, N = Ashva, G = Gaja, M = Mantri, K = Raja, P = Padati
```

## 3. Pieces

| Piece | Move (captures the same way) | Sources |
|-------|------------------------------|---------|
| **Raja** (king) | 1 step in any of the 8 directions. No castling. **No knight leap** (§6.3) | [W-Chat], [ChessCom], [Wikibooks], [LAPOC], [WorldChess]; supporting [Murray p. 225], [Averbakh p. 37] |
| **Mantri** (minister) | 1 step diagonally | [W-Chat], [Handout], [ChessCom], [LAPOC], [Wikibooks], [Daou], [Murray p. 64 (Nilakantha)] |
| **Ratha** (chariot) | Any distance orthogonally; no jumping | [W-Chat], [Handout], [ChessCom], [LAPOC] |
| **Gaja** (elephant) | Exactly 2 squares diagonally, **jumping** over the middle square (§6.4) | [W-Chat], [Handout], [ChessCom], [LAPOC], [Daou], [Murray pp. 59–60] |
| **Ashva** (horse) | Knight move; jumps | [W-Chat], [Handout], [ChessCom], [LAPOC] |
| **Padati** (foot soldier) | 1 step straight forward; captures 1 step diagonally forward. **No double step, no en passant** | [W-Chat] (citing Pritchard 2007, p. 263), [Handout], [ChessCom], [LAPOC], [Daou] |
| **Promotion** | On reaching the last rank, a Padati becomes a **Mantri**, immediately and automatically. Multiple Mantris are allowed (majority, §6.2; team confirmed) | §6.2 |

## 4. Check and legality

- A move is illegal if it leaves your own Raja attacked. When in check, you must get out of check.
  [Handout], [Daou] ("goal is to Checkmate"), [W-Chat], [LAPOC].
- [ChessCom] and [WorldChess] describe winning by capturing the king ("no checkmate"). That is a
  minority view, so we don't follow it.

## 5. How a game ends

After each move by player **A** (player **B** is now to move), the engine checks these in order:

1. **Bare king:** B has only the Raja left → **A wins immediately.** B does not get to capture back
   for a draw (see §6.5).
2. **Checkmate:** B has no legal move and is in check → **A wins.**
3. **Stalemate:** B has no legal move and is not in check → **B wins** (see §6.1).
4. **Threefold repetition** → **draw.** Shatranj counted repetition as a draw [Murray p. 229].
5. **No-progress rule:** 140 plies (70 moves each) with no capture and no Padati move → **draw.** [AbsGames] (shatranj)
6. **Ply cap [ENG]:** `max_plies` reached → **draw.** Default 300; configurable.

The order between rules 1 and 3 is our own choice; no source covers the case of baring and
stalemating in the same move. Scoring is +1 for a win, −1 for a loss and 0 for a draw.

## 6. Contested rules — reference tally

### 6.1 Stalemate → **stalemated player wins** (team confirmed)
| Outcome | Counted sources | # |
|---|---|---|
| **Stalemated player wins** | [Murray pp. 60–61] (al-Adli: "old Indian rule"), [W-Chat], [Handout], [LAPOC] ("the player who stalemates the enemy King loses"), [Ludoteka] ("the stalemated king gets the victory"), [Daou] ("the player that Stalemates their opponent loses"), [WorldChess] | **7** |
| Stalemating player wins | [ChessCom], [Wikibooks], [ChOnline] | **3** |
| Draw | — | **0** |

Supporting evidence goes the other way: shatranj gave the win to the stalemating player [Murray p. 229], [W-Shat], [Daou §II].
[Murray] explicitly calls the Indian rule different. Nilakantha [Murray p. 64] is unclear: it lets a
stalemated king capture the piece that confines him.

### 6.2 Promotion → **always Mantri** (majority; team confirmed)
| Rule | Counted sources | # |
|---|---|---|
| **Always Mantri** | [Murray p. 64] (Nilakantha: "when it arrives at the last square, it becomes a Counsellor"), [W-Promo] "some sources", [Wikibooks] ("only … promote to mantris"), [ChOnline] ("promote only to … Mantri"); *indirect:* chessvariants.com (cited by W-Promo) | **4** (+1 indirect) |
| **Piece that started on that file** | [Handout] (automatic), [LAPOC] (no promotion on the king's square), [Ludoteka] (only to already-captured pieces; the Padati waits otherwise), [Daou] (only to already-captured pieces) | **4** |
| Undecided | [ChessCom], [WorldChess] ("pawns may have always promoted to a Mantri, or possibly…") | — |

Why the tie breaks toward Mantri:
- **Oldest two-player Indian evidence.** Nilakantha (c. 1600–1700) is the only pre-modern Sanskrit
  source on the two-player game here, and it says Counsellor.
- **Where the file-based rule comes from.** The documented file-based rules come from the 4-player
  dice game [Murray p. 72] and 19th-c. Hindustani chess [Murray p. 81], [W-Ind]. [Daou]'s own footnote 5
  says his rule is modelled on "19th century Indian chess variants".
- **Consistency.** The four file-based sources give three *different* versions (automatic; no
  king's-square promotion; captured pieces only). All Mantri-only sources agree.
- **Successor game.** Shatranj used Ferz-only promotion [Murray p. 226], [W-Shat], [Daou §II] (supporting).
- The file-based versions stay available behind a config flag.

### 6.3 Raja's one-time knight leap → **not included** (majority)
| Rule | Counted sources | # |
|---|---|---|
| **No leap** | [W-Chat], [ChessCom], [Wikibooks], [LAPOC] ("moves like a chess King"), [WorldChess] ("moved exactly like today's king") | **5** |
| One leap per game | [Handout], [Ludoteka], [Daou] | **3** |

[Murray p. 81] documents the leap as a 19th-century Hindustani chess rule. That is also the rule set
[Ludoteka] and [Daou] follow.

### 6.4 Gaja move → **alfil (2-square diagonal jump)** (clear majority)
| Move | Counted sources | # |
|---|---|---|
| **2 diagonal, jumping** | [W-Chat], [Handout], [ChessCom], [LAPOC], [Daou], [Murray pp. 59–60] ("widest spread … probable that it is the oldest move"), plus the project proposal | **6** |
| 1 forward or 1 diagonal (al-Biruni) | Listed only as an alternative, in [W-Chat], [Murray], [Daou] | 0 as primary |
| 2 orthogonal, jumping (al-Adli) | Listed only as an alternative, in [W-Chat], [Murray], [Daou] | 0 as primary |

### 6.5 Bare king → **baring side wins immediately** (majority)
| Rule | Counted sources | # |
|---|---|---|
| **Baring wins, no exception** | [Murray p. 61] ("Indian … players counted this a win to the first player"), [W-Chat "Additional rules"] (al-Adli), [ChOnline] ("an immediate win"), [LAPOC], [Daou] ("Bare King counts as a Win". His shatranj section adds the exception; the chaturanga section does not), [WorldChess] | **6** |
| Bared back → draw | [W-Chat "Rules"], [Wikibooks] | **2** |

W-Chat's "Rules" paragraph cites [LAPOC] for the draw exception, but [LAPOC] does not state it.
The exception is the Persian/Arabic shatranj rule [Murray p. 229]. Nilakantha counts a bare king
as a half-victory [Murray p. 64], but no other source follows that.

### 6.6 Setup → **crosswise (Raja e1 / d8)** (majority; team confirmed)
| Setup | Counted sources | # |
|---|---|---|
| **Crosswise** (Rajas on different files) | [W-Chat], [Handout] (counted with W-Chat, same diagram), [Wikibooks], [Ludoteka], [Daou], [WorldChess] | **5** |
| **Facing** (Rajas on the same file) | [Murray pp. 27, 65, 80, 224] | **1** |

**Caveat to record in the report.** Murray is the only source that looks at primary evidence. He
finds the Rajas on the same file in Nilakantha (p. 65), in a Delhi manuscript of 1612 (p. 80) and
in the Arabic shatranj manuscripts. He says the manuscripts "oppose the antiquity of the modern
Asiatic crosswise arrangement" (p. 224), and his earliest reference to crosswise is from 1809 (p. 80).
None of the five crosswise sources cites evidence older than the 19th century. By the counting
method, crosswise wins. The facing setup stays available as a config option.

### 6.7 Draw rules → **included**
[Handout] says there are no repetition or 50-move draws. But shatranj had repetition and
perpetual-check draws [Murray p. 229] and a 70-move rule [AbsGames]. Self-play needs guaranteed
termination, so we keep rules 5.4–5.6.

## 7. Implications for the engine and encoding

- **Canonical board:** the crosswise setup is symmetric under a 180° **rotation**, so the board is
  rotated when Black is to move and the network always "plays White".
  - The facing-kings option uses a vertical mirror instead.
- **Input planes (15)**, implemented in `chaturanga/encoding.py`:
  - 12 piece planes (6 for the side to move, 6 for the opponent)
  - 1 repetition plane (the current position has occurred before). A second repetition plane
    would only ever be set in positions that are already drawn, so it is not needed.
  - 1 no-progress plane (plies since the last capture or Padati move, divided by 140)
  - 1 game-length plane (ply divided by `max_plies`), because the ply cap decides the result
- **Action space:** 64×64 = 4,096 from/to indices, with illegal indices masked. Promotion is implicit.
- **Repetition detection:** Zobrist hashing of (board, side to move).
- **Config flags** (`chaturanga/config.py`): `setup = crosswise | facing`,
  `promotion = mantri | file | file_restricted`, `stalemate = stalemated_wins | stalemating_wins | draw`.
  - `file_restricted` simplifies Murray p. 81: a Padati may promote to a piece type the side has
    lost at least one of. Murray's extra condition, that the lost Gaja must be the one able to
    reach the promotion square, is not modelled.

## 8. Unit tests implied (Milestone 1 checklist)

- Starting position: White has 16 legal moves (8 Padati, 4 Ashva, 4 Gaja).
- Gaja jumps over an occupied middle square and can only ever reach 8 squares.
- Padati: no double step; captures only diagonally forward; promotes to a Mantri on the last rank.
- Pinned pieces cannot move off the pin line; the Raja cannot step into check.
- Bare king ends the game at once, including when the bared side could have captured back.
- Checkmate; stalemate (the stalemated side wins); bare king takes priority over stalemate.
- Threefold repetition, the 140-ply no-progress draw and the ply cap.
- 180° rotation round-trip; move indices map consistently between orientations.
- Perft node counts at depths 1–4, cross-checked against a second, simpler move generator.
- Fuzz test: 10,000 random games with no crash, every game terminates, and every move played is in the legal list.
