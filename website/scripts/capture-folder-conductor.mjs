/**
 * Screenshot harness for "Folder is the conductor" (Settings > Developer >
 * Feature Previews).
 *
 * Runs the REAL built SPA (`website/dist`) behind the shared in-process static
 * server and answers every `/api/**` call from fixtures, so there is no gateway,
 * no kiro-cli and no live session involved — the frames are this branch's own
 * rendering of one fixed payload.
 *
 * The whole point of the change is a DELTA, and the delta is the preview flag,
 * so this script shoots the same payload twice in one run:
 *
 *   01-preview-off  `mc-preview-folder-conductor` absent — today's sidebar
 *   02-preview-on   the same flag at '1' — the conducted folder
 *
 * Shooting both here rather than running the script twice is deliberate: a
 * before/after pair taken from two runs can differ by anything that moved
 * between them, and the claim being made is specifically that NOTHING else did.
 *
 * The fixture seeds three folders, which are the three cases:
 *   - `remote-crew`     conducted: one session opened the other two, and one of
 *                       those is waiting on the user
 *   - `microvm-lane`    a sub-folder of it, conducted in its own right, so the
 *                       parent's badges are visibly a ROLLUP
 *   - `docs`            two hand-made sessions, nobody opened anybody, so this
 *                       folder must look identical in both frames
 *
 * Usage: node scripts/capture-folder-conductor.mjs [outDir]
 */
import { chromium } from 'playwright'
import { mkdirSync } from 'node:fs'
import { serveDist } from './lib/serve-dist.mjs'
import { logPageProblems, stubDashboardApi } from './lib/stub-dashboard-api.mjs'

const OUT = process.argv[2] || '../temp-screenshots/folder-conductor'
const PREVIEW_KEY = 'mc-preview-folder-conductor'

mkdirSync(OUT, { recursive: true })

const folders = [
  { id: 'f-remote', name: 'remote-crew', order: 0, collapsed: false },
  { id: 'f-microvm', name: 'microvm-lane', order: 0, collapsed: false, parent_id: 'f-remote' },
  { id: 'f-docs', name: 'docs', order: 1, collapsed: false },
]

const NOW = Date.now()
/** Re-base a fixture instant onto "now" so no row folds into the dormant
 *  expander, which would make a frame depend on the day it was taken. */
const at = (iso) =>
  new Date(NOW - (Date.parse('2026-10-09T16:40:00Z') - Date.parse(iso))).toISOString()

const slot = (key, title, folder_id, lastTs, extra = {}) => ({
  key,
  title,
  messages: 6,
  running: false,
  agent: 'kirocrew-lead',
  created: '2026-10-09T15:00:00Z',
  last_ts: at(lastTs),
  last_turn_ts: at(lastTs),
  folder_id,
  ...extra,
})

/**
 * One armed goal loop, on the `remote-crew` conductor.
 *
 * The folder row's status line is the conductor's own, so the only way to see it
 * is to give that session something to say: this is what renders "Loop 12/40"
 * under `remote-crew`, while `microvm-lane`'s conductor carries an unanswered
 * ask and renders the accent "Needs your answer" instead. Between them the two
 * folder rows cover both branches of the line in one frame.
 */
const loops = [{
  id: 'loop-remote',
  slot_key: 's-lead',
  message: 'Keep the remote crew lane green',
  active: true,
  idle_secs: 180,
  max_cycles: 40,
  cycle_count: 12,
}]

const slots = [
  // remote-crew: the conductor plus two workers, one of them asking.
  slot('s-lead', 'Remote crew rollout', 'f-remote', '2026-10-09T16:41:00Z'),
  slot('s-fargate', 'Fargate task definition', 'f-remote', '2026-10-09T16:38:00Z',
    { parent: { slot: 's-lead', key: 's-lead' } }),
  slot('s-token', 'Rotate remote crew token', 'f-remote', '2026-10-09T16:22:00Z',
    { parent: { slot: 's-lead', key: 's-lead' }, needs_input: true }),
  // microvm-lane: conducted in its own right, asking the user, one worker running.
  slot('s-micro', 'MicroVM lane', 'f-microvm', '2026-10-09T16:33:00Z', { needs_input: true }),
  slot('s-firecracker', 'Trim Firecracker image', 'f-microvm', '2026-10-09T16:31:00Z',
    { parent: { slot: 's-micro', key: 's-micro' }, running: true }),
  slot('s-snapshot', 'Snapshot restore latency baseline', 'f-microvm', '2026-10-09T16:12:00Z',
    { parent: { slot: 's-micro', key: 's-micro' } }),
  slot('s-vsock', 'vsock heartbeat', 'f-microvm', '2026-10-09T16:05:00Z',
    { parent: { slot: 's-micro', key: 's-micro' } }),
  // docs: no lineage at all, so no conductor — the control folder.
  slot('s-blog', 'Crew Log blog draft', 'f-docs', '2026-10-09T16:18:00Z'),
  slot('s-notes', 'Release notes', 'f-docs', '2026-10-09T15:55:00Z'),
]

async function shoot(base, name, { previewOn, hover = false }) {
  const browser = await chromium.launch()
  const context = await browser.newContext({
    viewport: { width: 1400, height: 1100 },
    // 12-13px sidebar type renders soft at 1x on GitHub.
    deviceScaleFactor: 2,
  })
  const page = await context.newPage()
  // Through the stub's own `localStorageEntries`, not a second init script:
  // Playwright does not order separately registered init scripts, so seeding
  // the flag elsewhere races the stub's storage clear.
  await stubDashboardApi(page, {
    folders,
    slots,
    localStorageEntries: previewOn ? { [PREVIEW_KEY]: '1' } : null,
    extra: async (path, route) => {
      if (path !== '/api/autonudge') return false
      await route.fulfill({ json: { enabled: true, loops } })
      return true
    },
  })
  logPageProblems(page)
  await page.goto(base + '/chat', { waitUntil: 'domcontentloaded' })
  await page.waitForTimeout(2600)
  if (hover) {
    // The Chat button is hover-only, exactly like the two controls beside it,
    // so it needs its own frame: in the resting frame that space holds the
    // row's badges, which is the state every row the pointer is not on shows.
    await page.locator('[data-folder-row="f-remote"]').hover()
    await page.waitForTimeout(400)
  } else {
    // Park the pointer off the list so no row is accidentally hovered.
    await page.mouse.move(1200, 900)
    await page.waitForTimeout(200)
  }
  // Fixed crop anchored to the sidebar panel, never to a row: a row-anchored
  // crop moves with the very layout being compared and hides the delta.
  await page.screenshot({
    path: `${OUT}/${name}.png`,
    clip: { x: 200, y: 110, width: 380, height: 700 },
  })
  console.log('wrote', `${OUT}/${name}.png`)
  await browser.close()
}

async function main() {
  const { srv, base } = await serveDist()
  await shoot(base, '01-preview-off', { previewOn: false })
  await shoot(base, '02-preview-on', { previewOn: true })
  await shoot(base, '03-preview-on-hover', { previewOn: true, hover: true })
  srv.close()
}

main().catch(err => { console.error(err); process.exit(1) })
