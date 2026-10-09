/**
 * Conductor detection, stated as cases rather than inferred from a render.
 *
 * The detection rule decides whether a folder reshapes itself, so the cases that
 * must NOT resolve to a conductor matter as much as the ones that must: a false
 * positive hides a session's card behind a one-line worker row, and a reader
 * looking for that session would have no way to tell the row apart from a bug.
 */
import { describe, it, expect } from 'vitest'

import { detectFolderConductor, rollupFolderConductor, type FolderConductorRow } from './folderConductor'

/** A row, written as the three fields the detector reads. */
const row = (key: string, parent?: string | null, created?: string): FolderConductorRow => ({
  key,
  folder_id: 'f1',
  parent: parent === undefined ? null : { slot: parent ?? undefined, key: parent },
  created,
})

describe('detectFolderConductor', () => {
  it('names the row that opened the others', () => {
    const found = detectFolderConductor([row('lead'), row('w1', 'lead'), row('w2', 'lead')])
    expect(found).toEqual({ conductorKey: 'lead', workerKeys: ['w1', 'w2'] })
  })

  it('reaches a worker nested below another worker', () => {
    // The folder's workers are its whole in-folder subtree, not just the
    // conductor's direct children: a helper a worker opened is still something
    // the conductor's run produced, and leaving it out would draw it as a
    // stray card beside the compact rows.
    const found = detectFolderConductor([row('lead'), row('w1', 'lead'), row('w1a', 'w1')])
    expect(found?.workerKeys).toEqual(['w1', 'w1a'])
  })

  it('returns workers in the order the rows arrived, not discovery order', () => {
    // Order is INHERITED from the lane that sorted the rows. Returning them in
    // the order the walk happened to reach them would make the folder disagree
    // with every other lane about the same two sessions.
    const found = detectFolderConductor([row('lead'), row('w1a', 'w1'), row('w1', 'lead')])
    expect(found?.workerKeys).toEqual(['w1a', 'w1'])
  })

  it('picks the TOP of the folder lineage, not a worker that dispatched helpers', () => {
    const found = detectFolderConductor([row('lead'), row('w1', 'lead'), row('w1a', 'w1')])
    expect(found?.conductorKey).toBe('lead')
  })

  it('ignores a creator that is not filed in this folder', () => {
    // The rows handed in are one folder's. A row whose creator lives elsewhere
    // is a ROOT here, so it can still conduct this folder.
    const rows = [
      { key: 'lead', folder_id: 'f1', parent: { slot: 'outside', key: 'outside' } },
      row('w1', 'lead'),
    ]
    expect(detectFolderConductor(rows)?.conductorKey).toBe('lead')
  })

  it('finds no conductor when nobody opened anybody', () => {
    // The ordinary folder: hand-made sessions sitting side by side. It must
    // render exactly as it does today, so the detector has to say null rather
    // than pick the first row.
    expect(detectFolderConductor([row('a'), row('b'), row('c')])).toBeNull()
  })

  it('finds no conductor in a folder holding one session', () => {
    expect(detectFolderConductor([row('solo')])).toBeNull()
  })

  it('finds no conductor in an empty folder', () => {
    expect(detectFolderConductor([])).toBeNull()
  })

  it('refuses a citation whose creator is not running', () => {
    // `parent.key` is null when the creator is not in the payload. The row
    // still cites it in `parent.slot`, and that citation must not establish a
    // conductor the user cannot open.
    const rows: FolderConductorRow[] = [
      { key: 'w1', folder_id: 'f1', parent: { slot: 'gone', key: null } },
      { key: 'w2', folder_id: 'f1', parent: { slot: 'gone', key: null } },
    ]
    expect(detectFolderConductor(rows)).toBeNull()
  })

  it('refuses a row that cites itself', () => {
    expect(detectFolderConductor([row('a', 'a'), row('b')])).toBeNull()
  })

  it('terminates on a lineage cycle instead of spinning', () => {
    // Our backend already refuses a cycle, so this is a payload we do not
    // produce. A view must still paint: every member of the cycle has an
    // in-folder creator, so none of them is a candidate and the answer is null.
    const rows = [row('a', 'b'), row('b', 'a')]
    expect(detectFolderConductor(rows)).toBeNull()
  })

  it('still answers when a cycle hangs below a real root', () => {
    // The root is a candidate; the walk below it must not revisit the cycle.
    const rows = [row('lead'), row('x', 'lead'), row('y', 'x'), row('x2', 'y')]
    const found = detectFolderConductor(rows)
    expect(found?.conductorKey).toBe('lead')
    expect(found?.workerKeys).toEqual(['x', 'y', 'x2'])
  })

  it('breaks a tie on worker count, then creation, then key', () => {
    const byCount = detectFolderConductor([
      row('a', null, '2026-01-01T00:00:00Z'), row('a1', 'a'),
      row('b', null, '2026-01-02T00:00:00Z'), row('b1', 'b'), row('b2', 'b'),
    ])
    expect(byCount?.conductorKey).toBe('b')

    const byCreated = detectFolderConductor([
      row('b', null, '2026-01-02T00:00:00Z'), row('b1', 'b'),
      row('a', null, '2026-01-01T00:00:00Z'), row('a1', 'a'),
    ])
    expect(byCreated?.conductorKey).toBe('a')

    const byKey = detectFolderConductor([row('b'), row('b1', 'b'), row('a'), row('a1', 'a')])
    expect(byKey?.conductorKey).toBe('a')
  })

  it('sorts a row with no creation instant last, not first', () => {
    // An absent `created` is unknown. Reading it as the epoch would let a
    // payload gap outrank a real timestamp.
    const found = detectFolderConductor([
      row('nostamp'), row('n1', 'nostamp'),
      row('stamped', null, '2026-06-01T00:00:00Z'), row('s1', 'stamped'),
    ])
    expect(found?.conductorKey).toBe('stamped')
  })

  it('does not depend on the order the rows are handed in', () => {
    const rows = [row('lead'), row('w1', 'lead'), row('w2', 'lead')]
    const forward = detectFolderConductor(rows)
    const reversed = detectFolderConductor([...rows].reverse())
    expect(reversed?.conductorKey).toBe(forward?.conductorKey)
  })

  it('leaves a row the conductor never opened out of the workers', () => {
    // It keeps its ordinary session card, so turning this rendering on cannot
    // hide a session from someone looking for it.
    const found = detectFolderConductor([row('lead'), row('w1', 'lead'), row('loose')])
    expect(found?.workerKeys).toEqual(['w1'])
  })
})

describe('rollupFolderConductor', () => {
  const rows: Record<string, FolderConductorRow[]> = {
    parent: [row('lead'), row('w1', 'lead')],
    child: [{ key: 'c1', folder_id: 'f2' }, { key: 'c2', folder_id: 'f2' }],
    grandchild: [{ key: 'g1', folder_id: 'f3' }],
  }
  const kids: Record<string, string[]> = { parent: ['child'], child: ['grandchild'] }
  const walk = {
    rowsIn: (id: string) => rows[id] ?? [],
    childFolderIds: (id: string) => kids[id] ?? [],
    wantsUser: (r: FolderConductorRow) => r.key === 'g1' || r.key === 'lead',
    isRunning: (r: FolderConductorRow) => r.key === 'w1',
  }

  it('counts the whole subtree, so a collapsed row hides nothing', () => {
    expect(rollupFolderConductor('parent', walk))
      .toEqual({ sessions: 5, needsYou: 2, running: 1 })
  })

  it('counts a leaf folder as itself', () => {
    expect(rollupFolderConductor('grandchild', walk))
      .toEqual({ sessions: 1, needsYou: 1, running: 0 })
  })

  it('counts an unknown folder as empty rather than throwing', () => {
    expect(rollupFolderConductor('nope', walk)).toEqual({ sessions: 0, needsYou: 0, running: 0 })
  })

  it('counts a folder cycle once instead of spinning', () => {
    const loop = {
      ...walk,
      rowsIn: (id: string) => (id === 'a' || id === 'b' ? [{ key: `${id}1`, folder_id: id }] : []),
      childFolderIds: (id: string) => (id === 'a' ? ['b'] : id === 'b' ? ['a'] : []),
      wantsUser: () => false,
      isRunning: () => false,
    }
    expect(rollupFolderConductor('a', loop).sessions).toBe(2)
  })
})
