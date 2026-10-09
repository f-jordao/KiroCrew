/**
 * Screenshot harness for the default custom agent bar on the Custom agents tab
 * (#18411, option A): one bar above the list that is both the readout and the
 * control for `agent.default_agent`.
 *
 * Runs the REAL built SPA (website/dist) behind the shared in-process static
 * server and answers every /api/** call from fixtures via Playwright route
 * interception — gateway-free, no kiro-cli, no dashboard auth.
 *
 * Frames, each in dark and light:
 *   bar        the tab with the bar showing the EFFECTIVE default (`kirocrew`,
 *              the runtime's own template, while the stored value is "")
 *   open       the picker open: a package template is offered, a crewmate's
 *              private copy is not
 *   picked     after choosing `atlas`: the bar reads `atlas` and the row's
 *              usage line carries "Default custom agent"
 *   refused    the server's 409 for a private copy, reported IN the bar
 *
 * SELF-CHECKS (throw = no stale frame): the bar's trigger text, the open
 * list's membership, the PUT body, the usage-line marker, the inline error, and
 * that no dialog is open when a frame is taken.
 *
 * Usage: node scripts/capture-default-template-bar.mjs [outDir] [prefix]
 */
import { chromium } from 'playwright'
import { mkdirSync } from 'node:fs'
import { serveDist } from './lib/serve-dist.mjs'
import { KIROCREW_CONFIG_FIXTURE, json, logPageProblems, stubDashboardApi } from './lib/stub-dashboard-api.mjs'

const OUT = process.argv[2] || '../temp-screenshots/default-template-bar'
const PREFIX = process.argv[3] || 'after'
mkdirSync(OUT, { recursive: true })

const BAR_LABEL = 'Custom agent for new chats that don’t pick one'

const CREWS = [
  { name: 'default', kiro_agent: 'kirocrew', workspace: 'default', memory_store: 'default', description: 'Used for all new chats', source: 'user', model: '', triggers: '', session_color: '' },
  { name: 'pr-bot', kiro_agent: 'pr-bot-copy', workspace: 'default', memory_store: 'default', description: 'Reviews pull requests', source: 'user', model: '', triggers: '', session_color: '' },
]

const tmpl = over => ({
  name: 'x', filename: 'x.json', description: '', model: '', skills: [], mcp_servers: [],
  source: 'builtin', package: '', scope: 'global', kirocrew_owned: false, forked_from: '', private_to: '',
  read_only: null, default_eligible: true, used_by: [], ...over,
})
const defaultRef = { kind: 'default', id: 'default', label: 'default agent' }
const templates = (defaultName) => [
  tmpl({ name: 'reviewer', filename: 'reviewer.json', description: 'Careful code reviewer', model: 'claude-opus-4.8', used_by: defaultName === 'reviewer' ? [defaultRef] : [] }),
  tmpl({ name: 'atlas', filename: 'atlas.json', description: 'Long-horizon planner', used_by: defaultName === 'atlas' ? [defaultRef] : [] }),
  tmpl({ name: 'papyrus-writer', filename: 'papyrus-papyrus-writer.json', description: 'LaTeX co-author', source: 'package', package: 'papyrus', read_only: 'package' }),
  tmpl({ name: 'pr-bot-copy', filename: 'pr-bot-copy.json', description: 'Careful code reviewer', forked_from: 'reviewer', private_to: 'pr-bot', read_only: 'private_copy', default_eligible: false, used_by: [{ kind: 'crew', id: 'pr-bot', label: 'pr-bot' }] }),
  tmpl({ name: 'kirocrew', filename: 'kirocrew.json', description: 'Built-in', source: 'kirocrew', kirocrew_owned: true, read_only: 'runtime', used_by: [{ kind: 'crew', id: 'default', label: 'default' }, ...(defaultName === 'kirocrew' ? [defaultRef] : [])] }),
]

const DETAIL = name => ({
  name, model: '', description: templates('').find(t => t.name === name)?.description || '',
  prompt: 'You are a careful reviewer. Read before you judge.', skills: [], tools: ['fs_read', 'fs_write', 'execute_bash'],
  allowedTools: ['fs_read'], resources: [], mcpServers: { 'kirocrew-core': {} },
})

const CFG = {
  ...KIROCREW_CONFIG_FIXTURE,
  agents: Object.fromEntries(CREWS.map(c => [c.name, { kiro_agent: c.kiro_agent, workspace: c.workspace, memory_store: c.memory_store, description: c.description, source: 'config' }])),
  default_agent: 'default',
  workspaces: { default: { dir: '~/.kiro/crew/workspace' } },
  memory_stores: { default: { description: 'Shared memory', embedding_provider: 'bge-m3' } },
}

/** Per-context server state: the stored default and the PUT bodies seen. */
function makeExtra(state) {
  return async function extra(path, route) {
    const method = route.request().method()
    if (path === '/api/config/kirocrew') return json(route, CFG), true
    if (path === '/api/agents') return json(route, { agents: CREWS, default_agent: 'default' }), true
    if (path === '/api/agents/installed') return json(route, []), true
    if (path === '/api/agents/templates') return json(route, { templates: templates(state.stored || 'kirocrew') }), true
    if (path === '/api/config/default-template' && method === 'PUT') {
      const body = route.request().postDataJSON() || {}
      state.puts.push(body.template)
      if (body.template === 'pr-bot-copy') {
        return json(route, { error: "Template 'pr-bot-copy' is crewmate 'pr-bot's private copy; it cannot be the default template.", code: 'template_private_copy' }, 409), true
      }
      state.stored = body.template
      return json(route, { ok: true, default_template: body.template, effective: body.template || 'kirocrew' }), true
    }
    if (path === '/api/config/default-template') return json(route, { default_template: state.stored, effective: state.stored || 'kirocrew' }), true
    if (path.startsWith('/api/agents/detail/')) return json(route, DETAIL(decodeURIComponent(path.split('/api/agents/detail/')[1] || ''))), true
    if (path === '/api/mcp' || path === '/api/mcp/probe') return json(route, []), true
    if (path === '/api/connections/status') return json(route, { schema_version: 1, connections: [] }), true
    if (path === '/api/workspaces') return json(route, { workspaces: [{ name: 'default' }] }), true
    if (path === '/api/skills' || path === '/api/skills/catalog') return json(route, []), true
    if (path === '/api/models') return json(route, { models: [{ name: 'claude-opus-4.8' }] }), true
    return false
  }
}

async function noDialog(page, where) {
  const n = await page.getByRole('dialog').count()
  if (n !== 0) throw new Error(`${where}: ${n} dialog(s) open on top of the frame`)
}

const { srv, base } = await serveDist()
const browser = await chromium.launch()
const wrote = []

try {
  for (const theme of ['dark', 'light']) {
    const state = { stored: '', puts: [] }
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2, baseURL: base })
    const page = await ctx.newPage()
    logPageProblems(page)
    await stubDashboardApi(page, { theme, extra: makeExtra(state) })

    await page.goto('/capabilities?tab=templates', { waitUntil: 'domcontentloaded' })
    const main = page.locator('#main-content')
    const bar = main.getByTestId('default-template-bar')
    await bar.waitFor({ state: 'visible', timeout: 20000 })
    const trigger = bar.getByRole('combobox', { name: BAR_LABEL })
    await trigger.waitFor({ state: 'visible', timeout: 15000 })
    // Stored "" renders as what actually starts, never as a blank or the first option.
    await page.waitForFunction(el => (el.textContent || '').includes('kirocrew'), await trigger.elementHandle(), { timeout: 15000 })
    await noDialog(page, `${theme} bar`)
    await page.waitForTimeout(400)
    let out = `${OUT}/${PREFIX}-${theme}-bar.png`
    await page.screenshot({ path: out }); wrote.push(out)

    // ── Open: eligibility is what the list shows ───────────────────────
    await trigger.click()
    const list = page.getByRole('listbox').filter({ hasNot: page.getByRole('option', { name: /Careful code reviewer/ }) }).last()
    await list.waitFor({ state: 'visible', timeout: 10000 })
    const names = await list.getByRole('option').allTextContents()
    for (const want of ['reviewer', 'atlas', 'papyrus-writer', 'kirocrew']) {
      if (!names.some(n => n.trim() === want)) throw new Error(`${theme} open: "${want}" missing from the picker (saw ${JSON.stringify(names)})`)
    }
    if (names.some(n => n.trim() === 'pr-bot-copy')) throw new Error(`${theme} open: a crewmate's private copy leaked into the picker`)
    await page.waitForTimeout(300)
    out = `${OUT}/${PREFIX}-${theme}-open.png`
    await page.screenshot({ path: out }); wrote.push(out)

    // ── Pick atlas: the PUT, the readout, the row marker ───────────────
    await list.getByRole('option', { name: 'atlas', exact: true }).click()
    await page.waitForFunction(el => (el.textContent || '').includes('atlas'), await trigger.elementHandle(), { timeout: 15000 })
    if (state.puts[0] !== 'atlas') throw new Error(`${theme} picked: PUT body was ${JSON.stringify(state.puts)}`)
    // The row itself carries a "Default" badge; the detail pane's usage line
    // names it in full. Both must move to atlas.
    const atlasRow = main.getByRole('option', { name: /^atlas\b/ })
    await atlasRow.getByText('Default', { exact: true }).waitFor({ state: 'visible', timeout: 15000 })
    if (await main.getByRole('option', { name: /^kirocrew\b/ }).getByText('Default', { exact: true }).count()) throw new Error(`${theme} picked: kirocrew row still badged Default`)
    await atlasRow.click()
    await main.getByText('Default custom agent', { exact: true }).first().waitFor({ state: 'visible', timeout: 15000 })
    await noDialog(page, `${theme} picked`)
    await page.waitForTimeout(400)
    out = `${OUT}/${PREFIX}-${theme}-picked.png`
    await page.screenshot({ path: out }); wrote.push(out)

    // ── Refused: a private copy via a stale list, reported in the bar ──
    // The picker never offers it, so the refusal is reached the way a stale
    // tab would reach it: the request is issued directly against the stub.
    await page.evaluate(() => fetch('/api/config/default-template', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ template: 'pr-bot-copy' }) }))
    if (state.puts[1] !== 'pr-bot-copy') throw new Error(`${theme} refused: stub never saw the private-copy PUT`)
    // And the UI path: the same refusal rendered where the control is, by
    // handing the 409 to the bar's own mutation through a visible pick.
    await page.route('**/api/config/default-template', async route => {
      if (route.request().method() !== 'PUT') return route.fallback()
      await json(route, { error: 'refused', code: 'template_private_copy' }, 409)
    })
    await trigger.click()
    const list2 = page.getByRole('listbox').filter({ hasNot: page.getByRole('option', { name: /Careful code reviewer/ }) }).last()
    await list2.waitFor({ state: 'visible', timeout: 10000 })
    await list2.getByRole('option', { name: 'reviewer', exact: true }).click()
    const err = bar.getByTestId('default-template-error')
    await err.waitFor({ state: 'visible', timeout: 15000 })
    if (!/can’t be the default/.test((await err.textContent()) || '')) throw new Error(`${theme} refused: inline error text not rendered`)
    await noDialog(page, `${theme} refused`)
    await page.waitForTimeout(400)
    out = `${OUT}/${PREFIX}-${theme}-refused.png`
    await page.screenshot({ path: out }); wrote.push(out)

    await ctx.close()
  }
} finally {
  await browser.close()
  srv.close()
}
for (const w of wrote) console.log(`wrote ${w}`)
