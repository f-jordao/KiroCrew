/**
 * Which session CONDUCTS a sidebar folder — derived, never stored.
 *
 * The question this answers: a folder often holds one session that opened every
 * other session in it (a conductor and its workers). Today the sidebar draws
 * that session as one more card inside the folder, so the folder row and the
 * conductor's card are two rows saying one thing, and the folder row itself says
 * nothing about whether the run inside needs the user.
 *
 * Everything here is a PURE function of data the sidebar already renders from:
 * `folder_id` (folder membership) and `parent.key` (who opened whom, the same
 * edge `sessionLineage.ts` nests the conductor lane by). No backend field, no new
 * API, no persisted "this folder's conductor" record — which is the point. A
 * stored pointer would be a second source of truth that goes stale the moment a
 * session closes or is moved between folders, and it would have to be migrated
 * for every folder that already exists.
 *
 * Deliberately structural over `Slot`: these functions take the three fields they
 * read, so the tests state a case as four literals instead of a slot payload, and
 * a change to `Slot` cannot silently change what counts as a conductor.
 */

/** The least a row must carry to be placed. A subset of the sidebar's `Slot`. */
export interface FolderConductorRow {
  key: string
  /** The folder this row is filed in, or absent when it is unfiled. */
  folder_id?: string
  /**
   * The creator, as the slots broadcast reports it. `key` is the creator's row
   * key in THIS payload and is null when the creator is not running — see
   * `Slot.parent`. Only `key` is read here: a citation naming a closed creator
   * cannot establish a conductor, because the conductor must be a row the user
   * can still open.
   */
  parent?: { slot?: string; key?: string | null; ancestor?: boolean } | null
  /** ISO creation instant, used only to break a tie deterministically. */
  created?: string
}

/** The verdict for one folder. */
export interface FolderConductor {
  /** The row key of the session that conducts this folder. */
  conductorKey: string
  /**
   * Its in-folder descendants, in the order the rows arrived.
   *
   * These are the rows the folder draws as the conductor's workers. A row filed
   * in the folder that does NOT descend from the conductor is absent here on
   * purpose: it keeps its ordinary session card, so turning this rendering on can
   * never hide a session from the person looking for it.
   */
  workerKeys: string[]
}

/** Rolled-up counts for a folder row's badges. */
export interface FolderConductorRollup {
  /** Sessions in this folder's whole subtree, the conductor included. */
  sessions: number
  /** Of those, how many are waiting on the user. */
  needsYou: number
  /** Of those, how many are running. */
  running: number
}

/** How deep the rollup will walk before it stops. Matches `renderFolderBlock`. */
const MAX_FOLDER_DEPTH = 10

/**
 * The conductor of a folder, from the folder's OWN rows, or null.
 *
 * A row conducts this folder when both hold:
 *
 *   1. it opened at least one other row filed in the same folder, and
 *   2. nothing filed in the same folder opened IT.
 *
 * Rule 1 is what makes the claim "this session manages the folder" true of a
 * particular row rather than of whichever row happens to look like an agent. Rule
 * 2 is what keeps a mid-level worker that dispatched its own helpers from
 * outranking the session above it: the conductor is the TOP of the folder's own
 * lineage, so one folder always resolves to at most one conductor.
 *
 * Neither rule consults the agent name. A folder is conducted because of what its
 * sessions DID, and an agent-name guess would be wrong in both directions — a
 * session running a conductor agent that has dispatched nothing yet is not
 * conducting anything the user can see, and a plain session that dispatched six
 * workers is.
 *
 * Ties (two independent roots each with workers) are decided by worker count,
 * then by the earlier `created`, then by `key`. The order never matters for
 * correctness — it exists so two renders of one payload agree.
 *
 * Returns null for a folder with no such row, and a folder with no conductor is
 * rendered EXACTLY as it is today. That is the fallback the whole feature rests
 * on: the ordinary folder is not a degraded case of a conducted one.
 */
export function detectFolderConductor<R extends FolderConductorRow>(
  rows: readonly R[],
): FolderConductor | null {
  if (rows.length < 2) return null
  const inFolder = new Set<string>()
  for (const row of rows) if (row.key) inFolder.add(row.key)

  /** This row's creator, but only when the creator is filed here too. */
  const localParent = (row: R): string | null => {
    const parent = row.parent?.key
    if (parent == null || parent === row.key || !inFolder.has(parent)) return null
    return parent
  }

  const childrenOf = new Map<string, string[]>()
  for (const row of rows) {
    if (!row.key) continue
    const parent = localParent(row)
    if (parent == null) continue
    const kids = childrenOf.get(parent)
    if (kids) kids.push(row.key)
    else childrenOf.set(parent, [row.key])
  }
  if (childrenOf.size === 0) return null

  const candidates = rows.filter(row =>
    !!row.key && localParent(row) == null && (childrenOf.get(row.key)?.length ?? 0) > 0)
  if (candidates.length === 0) return null

  /** Every in-folder row below *key*, in arrival order. */
  const descendants = (key: string): string[] => {
    const out: string[] = []
    const seen = new Set<string>([key])
    const queue = [key]
    while (queue.length > 0) {
      const next = queue.shift()!
      for (const kid of childrenOf.get(next) ?? []) {
        // `localParent` already refused a self-edge, and a longer cycle would
        // otherwise spin here: a row may be reached once.
        if (seen.has(kid)) continue
        seen.add(kid)
        out.push(kid)
        queue.push(kid)
      }
    }
    return out
  }

  let best: { row: R; workers: string[] } | null = null
  for (const row of candidates) {
    const workers = descendants(row.key)
    if (best == null || compareCandidates(row, workers, best.row, best.workers) < 0) {
      best = { row, workers }
    }
  }
  if (best == null) return null
  // Arrival order, not discovery order: the folder renders workers in the order
  // its lane already sorted them, exactly as `buildLineage` inherits order.
  const workerSet = new Set(best.workers)
  return {
    conductorKey: best.row.key,
    workerKeys: rows.filter(r => workerSet.has(r.key)).map(r => r.key),
  }
}

/** Negative when *a* should win. Total, so the result cannot depend on input order. */
function compareCandidates<R extends FolderConductorRow>(
  a: R, aWorkers: readonly string[], b: R, bWorkers: readonly string[],
): number {
  if (aWorkers.length !== bWorkers.length) return bWorkers.length - aWorkers.length
  const aCreated = a.created ?? ''
  const bCreated = b.created ?? ''
  // A row with no `created` sorts last rather than first: an absent instant is
  // unknown, and treating "" as the earliest would let a payload gap outrank a
  // real timestamp.
  if (aCreated !== bCreated) {
    if (aCreated === '') return 1
    if (bCreated === '') return -1
    return aCreated < bCreated ? -1 : 1
  }
  return a.key < b.key ? -1 : a.key > b.key ? 1 : 0
}

/**
 * The badge numbers for a conducted folder row: its whole subtree, rolled up.
 *
 * Subtree rather than the folder's own rows, because the row the user reads is
 * the only row on screen when the folder is collapsed — a needs-you sitting two
 * sub-folders down has nowhere else to appear. This is the same choice the
 * conductor lane makes for a collapsed row's aggregate.
 *
 * `wantsUser` and `isRunning` are handed in rather than computed here: they are
 * the predicates the session ROWS already render from, and a second definition
 * here would let a folder's badge disagree with the dots it stands for.
 */
export function rollupFolderConductor<R extends FolderConductorRow>(
  folderId: string,
  input: {
    rowsIn: (folderId: string) => readonly R[]
    childFolderIds: (folderId: string) => readonly string[]
    wantsUser: (row: R) => boolean
    isRunning: (row: R) => boolean
  },
): FolderConductorRollup {
  const out: FolderConductorRollup = { sessions: 0, needsYou: 0, running: 0 }
  const visited = new Set<string>()
  const walk = (id: string, depth: number) => {
    // Both guards are load-bearing: `visited` stops a folder tree whose
    // parent_id edges form a cycle, and the depth cap stops a chain longer than
    // the renderer itself will draw from being counted into a row nobody can
    // open.
    if (depth > MAX_FOLDER_DEPTH || visited.has(id)) return
    visited.add(id)
    for (const row of input.rowsIn(id)) {
      out.sessions += 1
      if (input.wantsUser(row)) out.needsYou += 1
      if (input.isRunning(row)) out.running += 1
    }
    for (const child of input.childFolderIds(id)) walk(child, depth + 1)
  }
  walk(folderId, 0)
  return out
}
