/**
 * "Folder is the conductor" — the preview, and the OFF path that must be today's.
 *
 * The feature: a folder whose sessions were all opened by one of them draws that
 * session AS the folder row — the loop mark in the icon slot, today's session
 * count and needs-you / running badges on the right, the conductor's own status
 * line under the name, a Chat button on hover — and its workers become one line
 * each instead of a card each. The conductor's card is not drawn inside, because
 * the folder row IS that card.
 *
 * Tens of thousands of people already read this list, so the FIRST contract here
 * is the one with no feature in it: with the flag off (the default), the folder
 * renders exactly as it did before — folder glyph, direct-child count, session
 * cards for every row including the conductor, and no control the row did not
 * have. Those assertions are not decoration; they are why the flag exists.
 *
 * `PREVIEW_FOLDER_CONDUCTOR` is a localStorage flag, so a test turns the feature
 * on by writing the key the shipped toggle writes, through `setPreviewFlag` —
 * never by reaching past it.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { render, fireEvent } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Provider } from 'react-redux'
import { MemoryRouter } from 'react-router-dom'
import { createTestStore } from './helpers'
import { ThemeProvider } from '../hooks/useTheme'
import { PREVIEW_FOLDER_CONDUCTOR, setPreviewFlag } from '../utils/previewFlags'
import type { ChatFolder, ChatSlot } from '../types'
import type { RootState } from '../store'

vi.mock('framer-motion', async () => {
  const React = await import('react')
  const FRAMER_PROPS = new Set([
    'layout', 'layoutId', 'layoutScroll', 'initial', 'animate', 'exit',
    'transition', 'variants', 'whileHover', 'whileTap', 'whileInView',
    'drag', 'dragConstraints', 'dragElastic', 'onAnimationComplete',
  ])
  const make = (tag: string) =>
    React.forwardRef((props: Record<string, unknown>, ref: React.Ref<unknown>) => {
      const clean: Record<string, unknown> = {}
      for (const k of Object.keys(props)) {
        if (k === 'children') continue
        if (k === 'layoutId') { clean['data-layout-id'] = props[k]; continue }
        if (FRAMER_PROPS.has(k)) continue
        clean[k] = props[k]
      }
      return React.createElement(tag, { ...clean, ref }, props.children as React.ReactNode)
    })
  const motion = new Proxy({}, { get: (_t, tag: string) => make(tag) })
  return {
    motion,
    AnimatePresence: ({ children }: { children?: React.ReactNode }) => React.createElement(React.Fragment, null, children),
    LayoutGroup: ({ children }: { children?: React.ReactNode }) => React.createElement(React.Fragment, null, children),
  }
})

vi.mock('../components/ProjectPicker', () => ({ default: () => null }))

vi.mock('../pages/chat/ChatSettings', () => ({
  loadChatConfig: () => ({ tagColumnsEnabled: false, hideEmptyFolderBody: false, confirmCloseSession: false }),
  saveChatConfig: vi.fn(),
}))

const mocks = vi.hoisted(() => ({ chatFolders: vi.fn() }))

vi.mock('../api/client', () => ({
  SEARCH_MIN_CHARS: 2,
  api: new Proxy(mocks as Record<string, unknown>, {
    get: (target, prop: string) => (prop in target ? target[prop] : vi.fn().mockResolvedValue([])),
  }),
}))

Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation((q: string) => ({
    matches: false, media: q, onchange: null,
    addListener: vi.fn(), removeListener: vi.fn(),
    addEventListener: vi.fn(), removeEventListener: vi.fn(), dispatchEvent: vi.fn(),
  })),
})

import ChatSidebar from '../pages/ChatSidebar'

const LANE = 'folder-lane'
const SUB = 'folder-sub'
const PLAIN = 'folder-plain'
const LEAD = 'k-lead'
const W1 = 'k-worker-1'
const W2 = 'k-worker-2'

const folders: ChatFolder[] = [
  { id: LANE, name: 'remote-crew', order: 0 },
  { id: SUB, name: 'microvm-lane', order: 1, parent_id: LANE },
  { id: PLAIN, name: 'docs', order: 2 },
]

/**
 * One conducted folder, one sub-folder under it, one ordinary folder.
 *
 * `k-worker-2` carries `needs_input`, so the rollup has something to count and
 * the worker row has a reason to draw its accent dot. The sub-folder's own
 * session is what proves the parent's count ROLLS UP rather than counting its
 * own rows: 3 here plus 1 there is 4.
 *
 * No timestamps anywhere, deliberately: a row with no activity instant is never
 * stale, so every row stays in the live list and no assertion depends on where
 * `now` happens to sit. A single old `created` on the conductor was enough to
 * sweep its card into the collapsed stale section, which reads exactly like the
 * feature having removed it.
 */
const slots: ChatSlot[] = [
  { key: LEAD, title: 'Remote crew rollout', messages: 9, running: false, folder_id: LANE },
  { key: W1, title: 'Fargate task definition', messages: 3, running: false, folder_id: LANE, parent: { slot: LEAD, key: LEAD } },
  { key: W2, title: 'Rotate remote crew token', messages: 2, running: false, needs_input: true, folder_id: LANE, parent: { slot: LEAD, key: LEAD } },
  { key: 'k-sub', title: 'Trim Firecracker image', messages: 1, running: false, folder_id: SUB },
  { key: 'k-plain-a', title: 'Blog draft', messages: 4, running: false, folder_id: PLAIN },
  { key: 'k-plain-b', title: 'Release notes', messages: 1, running: false, folder_id: PLAIN },
]

/** An armed goal loop, in the shape the store holds after the automation seed. */
const armedLoop = (slotKey: string, cycleCount: number, maxCycles: number) => ({
  [slotKey]: {
    kind: 'legacy_goal_loop' as const,
    id: `loop-${slotKey}`,
    slotKey,
    message: 'Keep the lane green',
    idleSecs: 180,
    maxCycles,
    cycleCount,
    active: true,
    lastFireAt: 0,
    nextDueAt: 0,
    maxRuntimeSecs: 0,
    stoppedReason: '',
    monitorOutcome: '',
    monitorKind: '',
  },
})

function renderSidebar(slotData: ChatSlot[] = slots, automations: Record<string, unknown> = {}) {
  const store = createTestStore({
    dashboard: {
      status: {}, connected: true, slots: slotData, approvalMode: 'normal',
      channelTrusted: false, refreshTrigger: 0, unreadSlots: [], updateProgress: null,
      subagentRunning: {}, subagentDetails: {}, subagentText: {},
      sessionDefaultColor: null, sessionColorsMode: 'tint', sessionColorsPalette: 'horizon', sessionColorsIntensity: 'clear',
    } as unknown as RootState['dashboard'],
    chat: { activeSlot: null, automations } as unknown as RootState['chat'],
  })
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  qc.setQueryData(['chat-tags'], [])
  qc.setQueryData(['tag-columns'], [])
  qc.setQueryData(['chat-folders'], folders)
  qc.setQueryData(['kirocrewConfig'], { dashboard: {} })
  mocks.chatFolders.mockImplementation(() => Promise.resolve(folders))
  return render(
    <QueryClientProvider client={qc}>
      <Provider store={store}>
        <ThemeProvider>
          <MemoryRouter>
            <ChatSidebar
              slots={slotData} activeSlot={null} unreadSlots={[]}
              history={[]} historyHasMore={false} defaultAgent="" installedAgents={[]}
            />
          </MemoryRouter>
        </ThemeProvider>
      </Provider>
    </QueryClientProvider>,
  )
}

/** The folder row's own box, the only element carrying `data-folder-row`. */
function rowOf(container: HTMLElement, folderId: string): HTMLElement {
  const row = container.querySelector(`[data-folder-row="${folderId}"]`)
  if (!row) throw new Error(`no folder row for ${folderId}`)
  return row as HTMLElement
}
/**
 * A session CARD, as the ordinary lanes draw it.
 *
 * `data-slot-key`, which sits on the outermost row element the card renders —
 * NOT `data-session-row`, which carries the row IDENTITY (origin-qualified for a
 * federated row) and so does not match a bare slot key. An absent-looking card
 * that is really a mis-spelled selector reads identically to the feature having
 * removed it, which is why `cardsAreFound` below asserts this selector works.
 */
function cardOf(container: HTMLElement, key: string): Element | null {
  return container.querySelector(`[data-slot-key="${key}"]`)
}

beforeEach(() => {
  localStorage.clear()
  mocks.chatFolders.mockImplementation(() => Promise.resolve(folders))
})
afterEach(() => vi.clearAllMocks())

describe('sidebar folders with the preview OFF (the default)', () => {
  it('draws the folder glyph, not the conductor mark', () => {
    const { container } = renderSidebar()
    // The glyph box is the same element either way; what it CONTAINS is the
    // difference, and the lucide shape is an <svg> while the mark is a <span>.
    const shape = container.querySelector(`[data-testid="folder-collapse-${LANE}-shape"]`)
    expect(shape?.tagName.toLowerCase()).toBe('svg')
  })

  it('counts the folder\'s own children, not its subtree', () => {
    const { container } = renderSidebar()
    // 3 sessions filed here + 1 child folder = 4 by today's rule, which happens
    // to equal the rollup's own 4 — so the assertion is on the ordinary folder,
    // where the two rules really do differ: 2 sessions, no child folder.
    expect(rowOf(container, PLAIN).textContent).toContain('2')
  })

  it('draws every session as a card, the conductor included', () => {
    const { container } = renderSidebar()
    expect(cardOf(container, LEAD)).not.toBeNull()
    expect(cardOf(container, W1)).not.toBeNull()
    expect(cardOf(container, W2)).not.toBeNull()
  })

  it('offers no Chat button, no badges and no status line', () => {
    const { container } = renderSidebar()
    expect(container.querySelector(`[data-testid="folder-conductor-chat-${LANE}"]`)).toBeNull()
    expect(container.querySelector(`[data-testid="folder-conductor-needs-you-${LANE}"]`)).toBeNull()
    expect(container.querySelector(`[data-testid="folder-conductor-line-${LANE}"]`)).toBeNull()
    expect(container.querySelector('[data-folder-conductor-worker]')).toBeNull()
  })
})

describe('sidebar folders with the preview ON', () => {
  beforeEach(() => { setPreviewFlag(PREVIEW_FOLDER_CONDUCTOR, true) })

  it('puts the conductor mark in the folder row\'s icon slot', () => {
    const { container } = renderSidebar()
    const shape = container.querySelector(`[data-testid="folder-collapse-${LANE}-shape"]`)
    // A <span> wrapping the mark, where the ordinary folder has the lucide
    // <svg> — and the disclosure chevron is still there, so clicking the row
    // still reads as expand/collapse.
    expect(shape?.tagName.toLowerCase()).toBe('span')
    expect(container.querySelector(`[data-testid="folder-collapse-${LANE}-disclosure"]`)).not.toBeNull()
  })

  it('still finds a card for a session this feature does not touch', () => {
    // The POSITIVE CONTROL for `cardOf`. Every "the card is gone" assertion
    // below is an absence, and an absence proves nothing until the selector is
    // known to find a card that is really there.
    const { container } = renderSidebar()
    expect(cardOf(container, 'k-plain-a')).not.toBeNull()
  })

  it('stops drawing the conductor\'s own card inside the folder', () => {
    const { container } = renderSidebar()
    // The whole point: the folder row is now that card, and two of them would
    // be one session drawn twice.
    expect(cardOf(container, LEAD)).toBeNull()
  })

  it('draws the workers as one-line rows instead of cards', () => {
    const { container } = renderSidebar()
    expect(cardOf(container, W1)).toBeNull()
    expect(cardOf(container, W2)).toBeNull()
    const w1 = container.querySelector(`[data-testid="folder-conductor-worker-${W1}"]`)
    expect(w1).not.toBeNull()
    expect(w1!.textContent).toContain('Fargate task definition')
  })

  it('rolls the subtree up into the row\'s count and needs-you badge', () => {
    const { container } = renderSidebar()
    const row = rowOf(container, LANE)
    // 3 sessions here + 1 in the sub-folder. A needs-you two rows down has
    // nowhere else to appear while the folder is collapsed, which is why the
    // badge counts the subtree rather than the direct children.
    expect(row.textContent).toContain('4')
    const badge = container.querySelector(`[data-testid="folder-conductor-needs-you-${LANE}"]`)
    expect(badge).not.toBeNull()
    expect(badge!.textContent).toContain('1')
  })

  it('shows the conductor\'s own status line under the name', () => {
    const { container } = renderSidebar()
    // Nothing to report on this conductor (no ask, no loop armed), so the line
    // renders nothing rather than a row saying "idle".
    expect(container.querySelector(`[data-testid="folder-conductor-status-${LEAD}"]`)).toBeNull()

    // Give the conductor the ask and the line says so, in the words its own
    // session card would have used.
    const asking = slots.map(s => (s.key === LEAD ? { ...s, needs_input: true } : s))
    const second = renderSidebar(asking)
    const line = second.container.querySelector(`[data-testid="folder-conductor-status-${LEAD}"]`)
    expect(line).not.toBeNull()
    expect(line!.textContent).toContain('Needs your answer')
  })

  it('says Loop N/M while the conductor\'s own loop is armed', () => {
    // The other branch of the status line, and the one no screenshot in this
    // branch can show: the loop record only reaches the store through the
    // automation seed on the websocket, which the capture harness swallows. The
    // numbers come from the conductor's record, in the session card's own words.
    const { container } = renderSidebar(slots, armedLoop(LEAD, 12, 40))
    const line = container.querySelector(`[data-testid="folder-conductor-status-${LEAD}"]`)
    expect(line).not.toBeNull()
    expect(line!.textContent).toContain('Loop 12/40')
  })

  it('lets the conductor\'s unanswered ask outrank its own loop', () => {
    // A looping conductor parked on a question is waiting on the user, not
    // working, and the row must not report progress it is not making.
    const asking = slots.map(s => (s.key === LEAD ? { ...s, needs_input: true } : s))
    const { container } = renderSidebar(asking, armedLoop(LEAD, 12, 40))
    const line = container.querySelector(`[data-testid="folder-conductor-status-${LEAD}"]`)
    expect(line!.textContent).toContain('Needs your answer')
    expect(line!.textContent).not.toContain('Loop 12/40')
  })

  it('opens the conductor\'s chat from the row\'s Chat button', () => {
    const { container } = renderSidebar()
    const chat = container.querySelector(`[data-testid="folder-conductor-chat-${LANE}"]`) as HTMLElement
    expect(chat).not.toBeNull()
    expect(chat.textContent).toContain('Chat')
    // The row's collapse state must not change: the Chat button is a sibling of
    // the collapse toggle, not a second gesture on it.
    const shell = container.querySelector(`[data-testid="folder-collapse-${LANE}"]`)?.parentElement as HTMLElement
    const before = shell.getAttribute('aria-expanded')
    fireEvent.click(chat)
    expect(shell.getAttribute('aria-expanded')).toBe(before)
  })

  it('leaves a folder with no conductor exactly as it was', () => {
    const { container } = renderSidebar()
    // `docs` holds two hand-made sessions: nobody opened anybody, so there is no
    // conductor and the folder is the ordinary one in every respect.
    const shape = container.querySelector(`[data-testid="folder-collapse-${PLAIN}-shape"]`)
    expect(shape?.tagName.toLowerCase()).toBe('svg')
    expect(rowOf(container, PLAIN).textContent).toContain('2')
    expect(cardOf(container, 'k-plain-a')).not.toBeNull()
    expect(container.querySelector(`[data-testid="folder-conductor-chat-${PLAIN}"]`)).toBeNull()
  })

  it('keeps a session the conductor never opened as a card', () => {
    // Turning the preview on must not be able to hide a session from someone
    // looking for it, so a row that does not descend from the conductor keeps
    // the card it has always had.
    const withStray = [...slots, { key: 'k-stray', title: 'Hand-made note', messages: 1, running: false, folder_id: LANE } as ChatSlot]
    const { container } = renderSidebar(withStray)
    expect(cardOf(container, 'k-stray')).not.toBeNull()
    expect(container.querySelector('[data-testid="folder-conductor-worker-k-stray"]')).toBeNull()
  })
})
