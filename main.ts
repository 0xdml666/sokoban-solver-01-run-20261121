// Sokoban solver for make.fun's "Build the best Sokoban solver, 21 November 2026".
//
// Strategy (written from the competition rules, not copied from any solver):
// The generator builds each level from a one-wide hallway plus 1-4 rooms, each
// room opening onto the hallway through a single door, with 3-4 boxes and as
// many goals. A box pushed out of a room through its door can never come back,
// so in an optimal solution every box stays in its room and a level's
// push-optimal count is the sum of its rooms' optima. This program therefore
//   1. splits each level into its hallway and its rooms (connected components
//      of floor off the hallway row),
//   2. solves each room push-optimally on its own with A* over pushes
//      (state = box positions + the player's reachable region, admissible
//      min-cost-matching heuristic, static dead squares, 2x2 blocks and
//      frozen-box pruning),
//   3. stitches the room solutions together with plain walking moves along
//      the hallway.
// Only push count is scored, so walking moves are free. The search is capped
// by expansions, never by the clock, so two runs write identical bytes.
//
// Local run: mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Directions: up, down, left, right. A push is written as the upper-case letter.
const DIRS = [
  { dx: 0, dy: -1, ch: 'u' },
  { dx: 0, dy: 1, ch: 'd' },
  { dx: -1, dy: 0, ch: 'l' },
  { dx: 1, dy: 0, ch: 'r' },
];
const INF = 1_000_000_000;

type Level = {
  w: number;
  h: number;
  wall: boolean[];
  goals: number[];
  goalSet: boolean[];
  boxes: number[];
  player: number;
};

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall: boolean[] = new Array(w * h).fill(true);
  const goalSet: boolean[] = new Array(w * h).fill(false);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      wall[i] = ch === '#';
      if (ch === '.' || ch === '*' || ch === '+') {
        goalSet[i] = true;
        goals.push(i);
      }
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, goalSet, boxes, player };
}

// Breadth-first walk over floor cells (boxes are obstacles). Returns the move
// string from `from` to `to`, or null when unreachable. `cells` restricts the
// region when given (room-local walks); otherwise the whole board is allowed.
function walkPath(
  lv: Level,
  from: number,
  to: number,
  blocked: boolean[] | Set<number>,
  allowed?: boolean[],
): string | null {
  if (from === to) return '';
  const n = lv.w * lv.h;
  const prev = new Int32Array(n).fill(-1);
  const prevCh = new Array<string>(n).fill('');
  const q: number[] = [from];
  prev[from] = from;
  let head = 0;
  const isBlocked = (c: number): boolean =>
    Array.isArray(blocked) ? blocked[c] === true : blocked.has(c);
  while (head < q.length) {
    const c = q[head++]!;
    for (const d of DIRS) {
      const nx = (c % lv.w) + d.dx;
      const ny = Math.floor(c / lv.w) + d.dy;
      if (nx < 0 || nx >= lv.w || ny < 0 || ny >= lv.h) continue;
      const m = ny * lv.w + nx;
      if (lv.wall[m] || prev[m] !== -1 || isBlocked(m)) continue;
      if (allowed && !allowed[m]) continue;
      prev[m] = c;
      prevCh[m] = d.ch;
      if (m === to) {
        let s = '';
        let cur = m;
        while (cur !== from) {
          s = prevCh[cur]! + s;
          cur = prev[cur]!;
        }
        return s;
      }
      q.push(m);
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Room decomposition
// ---------------------------------------------------------------------------

type Room = {
  cells: number[]; // global indices of the room's floor cells
  doors: number[]; // room cells adjacent to the hallway
  entries: number[]; // hallway cells adjacent to a door
  boxes: number[];
  goals: number[];
};

// Split a level into rooms: connected components of floor cells off the
// hallway row (the row the player starts on). Returns null when the level
// does not fit that shape, so the caller can fall back to a whole-level solve.
function decompose(lv: Level): Room[] | null {
  const hallwayRow = Math.floor(lv.player / lv.w);
  const seen = new Array<boolean>(lv.w * lv.h).fill(false);
  const rooms: Room[] = [];
  const boxRoom = new Map<number, number>();
  for (let i = 0; i < lv.w * lv.h; i++) {
    if (lv.wall[i] || seen[i]) continue;
    const y = Math.floor(i / lv.w);
    if (y === hallwayRow) continue;
    const cells: number[] = [];
    const q: number[] = [i];
    seen[i] = true;
    while (q.length > 0) {
      const c = q.pop()!;
      cells.push(c);
      for (const d of DIRS) {
        const nx = (c % lv.w) + d.dx;
        const ny = Math.floor(c / lv.w) + d.dy;
        if (nx < 0 || nx >= lv.w || ny < 0 || ny >= lv.h) continue;
        const m = ny * lv.w + nx;
        if (lv.wall[m] || seen[m] || Math.floor(m / lv.w) === hallwayRow) continue;
        seen[m] = true;
        q.push(m);
      }
    }
    const doors: number[] = [];
    const entries: number[] = [];
    for (const c of cells) {
      for (const d of DIRS) {
        const nx = (c % lv.w) + d.dx;
        const ny = Math.floor(c / lv.w) + d.dy;
        if (nx < 0 || nx >= lv.w || ny < 0 || ny >= lv.h) continue;
        const m = ny * lv.w + nx;
        if (!lv.wall[m] && Math.floor(m / lv.w) === hallwayRow) {
          if (!doors.includes(c)) doors.push(c);
          if (!entries.includes(m)) entries.push(m);
        }
      }
    }
    const boxes = lv.boxes.filter((b) => cells.includes(b));
    const goals = lv.goals.filter((g) => cells.includes(g));
    const room: Room = { cells, doors, entries, boxes, goals };
    rooms.push(room);
    boxes.forEach((b) => boxRoom.set(b, rooms.length - 1));
  }
  // Every box must sit in exactly one room, every room with boxes needs a
  // door, and boxes and goals must match per room.
  if (lv.boxes.some((b) => !boxRoom.has(b))) return null;
  for (const r of rooms) {
    if (r.boxes.length > 0 && r.doors.length === 0) return null;
    if (r.boxes.length !== r.goals.length) return null;
  }
  return rooms;
}

// ---------------------------------------------------------------------------
// Push-space A* for one enclosed region (a room, or a whole level)
// ---------------------------------------------------------------------------

type PushProblem = {
  // Global cell indices the player may stand on.
  moveCells: number[];
  // Global cell indices boxes may occupy (a subset of moveCells).
  roomCells: number[];
  goals: number[];
  boxes: number[];
  startCell: number;
  // Door cells the player must be able to reach at the end (rooms only).
  doors: number[];
  requireExit: boolean;
};

type PushSolution = {
  moves: string;
  finalPlayer: number;
  finalBoxes: number[];
};

// Flood fill from `from` over moveCells, treating `blocked` cells as walls.
// Returns a boolean array over local indices.
function floodLocal(
  localOf: Map<number, number>,
  neighbours: number[][],
  fromLocal: number,
  blockedLocal: Set<number>,
): boolean[] {
  const reach = new Array<boolean>(neighbours.length).fill(false);
  const q: number[] = [fromLocal];
  reach[fromLocal] = true;
  let head = 0;
  while (head < q.length) {
    const c = q[head++]!;
    for (const m of neighbours[c]!) {
      if (m < 0 || reach[m] || blockedLocal.has(m)) continue;
      reach[m] = true;
      q.push(m);
    }
  }
  return reach;
}

function solvePush(p: PushProblem, weight: number, cap: number): PushSolution | null {
  const n = p.moveCells.length;
  const localOf = new Map<number, number>();
  p.moveCells.forEach((g, i) => localOf.set(g, i));
  const isRoom = new Array<boolean>(n).fill(false);
  for (const g of p.roomCells) isRoom[localOf.get(g)!] = true;
  const isGoal = new Array<boolean>(n).fill(false);
  for (const g of p.goals) isGoal[localOf.get(g)!] = true;
  const doorLocal = p.doors.map((g) => localOf.get(g)!).filter((x) => x !== undefined);
  const startLocal = localOf.get(p.startCell)!;

  // Neighbours per local cell, per direction (local index or -1).
  const neighbours: number[][] = [];
  const width = (p as unknown as { width: number }).width;
  for (let i = 0; i < n; i++) {
    const g = p.moveCells[i]!;
    const x = g % width;
    const y = Math.floor(g / width);
    const row: number[] = [];
    for (const d of DIRS) {
      const nx = x + d.dx;
      const ny = y + d.dy;
      if (nx < 0 || ny < 0) {
        row.push(-1);
        continue;
      }
      const m = ny * width + nx;
      const li = localOf.get(m);
      row.push(li === undefined ? -1 : li);
    }
    neighbours.push(row);
  }

  // All-pairs box-move distances over room cells (ignoring boxes/player).
  const dist = new Int32Array(n * n).fill(INF);
  for (let s = 0; s < n; s++) {
    if (!isRoom[s]) continue;
    dist[s * n + s] = 0;
    const q: number[] = [s];
    let head = 0;
    while (head < q.length) {
      const c = q[head++]!;
      for (const m of neighbours[c]!) {
        if (m < 0 || !isRoom[m] || dist[s * n + m] !== INF) continue;
        dist[s * n + m] = dist[s * n + c]! + 1;
        q.push(m);
      }
    }
  }

  // Static dead squares: a room cell is live when a box there can be pulled
  // (ignoring other boxes) onto a goal. Fixpoint from the goals outward.
  const live = new Array<boolean>(n).fill(false);
  for (let i = 0; i < n; i++) if (isGoal[i]) live[i] = true;
  let changed = true;
  while (changed) {
    changed = false;
    for (let c = 0; c < n; c++) {
      if (!isRoom[c] || live[c]) continue;
      for (let di = 0; di < 4; di++) {
        const dest = neighbours[c]![di]!;
        if (dest < 0 || !live[dest]) continue;
        const from = neighbours[c]![di ^ 1]!;
        if (from < 0) continue;
        // The player may already stand behind the box, having walked there
        // while c was empty and followed the box in, so a floor square
        // behind it is enough for the pull to be possible. (Flooding
        // reachability from the door instead wrongly kills 1-wide corridors:
        // with the box on c the far side is cut off from the door, yet the
        // player can already be there.)
        live[c] = true;
        changed = true;
        break;
      }
    }
  }

  const goalLocal = p.goals.map((g) => localOf.get(g)!).sort((a, b) => a - b);
  const boxLocal0 = p.boxes.map((g) => localOf.get(g)!).sort((a, b) => a - b);
  const m = boxLocal0.length;

  // Permutations of goal indices for the matching heuristic.
  const perms: number[][] = [];
  const genPerm = (arr: number[], k: number): void => {
    if (k === arr.length) {
      perms.push([...arr]);
      return;
    }
    for (let i = k; i < arr.length; i++) {
      [arr[k], arr[i]] = [arr[i]!, arr[k]!];
      genPerm(arr, k + 1);
      [arr[k], arr[i]] = [arr[i]!, arr[k]!];
    }
  };
  genPerm(
    goalLocal.map((_, i) => i),
    0,
  );

  const heuristic = (boxes: number[]): number => {
    let best = INF;
    for (const perm of perms) {
      let s = 0;
      let bad = false;
      for (let i = 0; i < m; i++) {
        const d = dist[boxes[i]! * n + goalLocal[perm[i]!]!]!;
        if (d >= INF) {
          bad = true;
          break;
        }
        s += d;
      }
      if (!bad && s < best) best = s;
    }
    return best;
  };

  const solved = (boxes: number[]): boolean =>
    boxes.every((b) => isGoal[b]);

  // Dynamic deadlock: 2x2 blocks of walls/boxes holding no goal, and boxes
  // frozen by walls or other frozen boxes that are not on goals.
  const deadlocked = (boxes: number[]): boolean => {
    const boxSet = new Set(boxes);
    for (const b of boxes) {
      if (!live[b]) return true;
    }
    // 2x2 blocks.
    for (const b of boxes) {
      const g = p.moveCells[b]!;
      const x = g % width;
      const y = Math.floor(g / width);
      for (const [ox, oy] of [
        [-1, -1],
        [-1, 0],
        [0, -1],
        [0, 0],
      ]) {
        let boxCount = 0;
        let allBlocked = true;
        let hasGoal = false;
        for (const [cx, cy] of [
          [0, 0],
          [1, 0],
          [0, 1],
          [1, 1],
        ]) {
          const gx = x + ox + cx;
          const gy = y + oy + cy;
          if (gx < 0 || gx >= width || gy < 0) continue;
          const cell = gy * width + gx;
          const li = localOf.get(cell);
          if (li === undefined) continue; // outside the region: wall
          if (isGoal[li]) hasGoal = true;
          if (boxSet.has(li)) boxCount++;
          else if (isRoom[li]) allBlocked = false;
        }
        if (allBlocked && boxCount >= 2 && !hasGoal) return true;
      }
    }
    // Frozen boxes (fixpoint: blocked by walls or frozen boxes only).
    const frozen = new Set<number>();
    let grew = true;
    while (grew) {
      grew = false;
      for (const b of boxes) {
        if (frozen.has(b)) continue;
        let stuck = true;
        for (let di = 0; di < 4; di++) {
          const dest = neighbours[b]![di]!;
          const from = neighbours[b]![di ^ 1]!;
          const destBlocked =
            dest < 0 || !isRoom[dest] || (boxSet.has(dest) && frozen.has(dest));
          const fromBlocked =
            from < 0 || (boxSet.has(from) && frozen.has(from));
          if (!destBlocked && !fromBlocked) {
            stuck = false;
            break;
          }
        }
        if (stuck) {
          frozen.add(b);
          grew = true;
        }
      }
    }
    for (const b of frozen) if (!isGoal[b]) return true;
    return false;
  };

  // State encoding: boxes sorted, each < BASE, plus canonical player < BASE.
  // Layout (least significant digit first): padding, boxes, player.
  const BASE = 512;
  const PAD = 511;
  const encode = (boxes: number[], playerCanon: number): number => {
    let k = playerCanon;
    for (let i = m - 1; i >= 0; i--) k = k * BASE + boxes[i]!;
    for (let i = m; i < 4; i++) k = k * BASE + PAD;
    return k;
  };
  const decodeBoxes = (key: number): number[] => {
    let k = key;
    const digits: number[] = [];
    for (let i = 0; i < 5; i++) {
      digits.push(k % BASE);
      k = Math.floor(k / BASE);
    }
    // digits[0] is the last value added (a pad, or box 0 when m == 4);
    // boxes occupy digits[4 - m .. 3], the player digit[4].
    const boxes: number[] = [];
    for (let i = 0; i < m; i++) boxes.push(digits[4 - m + i]!);
    return boxes.sort((a, b) => a - b);
  };

  // A* state storage.
  const stKey: number[] = [];
  const stG: number[] = [];
  const stF: number[] = [];
  const stParent: number[] = [];
  const stPushBox: number[] = [];
  const stPushDir: number[] = [];
  const stPlayerEnd: number[] = [];
  const best = new Map<number, number>();
  const heap: number[] = [];
  const heapLess = (a: number, b: number): boolean =>
    stF[a]! < stF[b]! ||
    (stF[a] === stF[b] && (stG[a]! > stG[b]! || (stG[a] === stG[b] && a < b)));
  const heapPush = (s: number): void => {
    let i = heap.length;
    heap.push(s);
    while (i > 0) {
      const par = (i - 1) >> 1;
      if (!heapLess(heap[i]!, heap[par]!)) break;
      [heap[i], heap[par]] = [heap[par]!, heap[i]!];
      i = par;
    }
  };
  const heapPop = (): number => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && heapLess(heap[c + 1]!, heap[c]!)) c++;
        if (!heapLess(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  const addState = (
    key: number,
    g: number,
    f: number,
    parent: number,
    pushBox: number,
    pushDir: number,
    playerEnd: number,
  ): number => {
    const idx = stKey.length;
    stKey.push(key);
    stG.push(g);
    stF.push(f);
    stParent.push(parent);
    stPushBox.push(pushBox);
    stPushDir.push(pushDir);
    stPlayerEnd.push(playerEnd);
    best.set(key, idx);
    heapPush(idx);
    return idx;
  };

  const fOf = (g: number, h: number): number =>
    weight === Infinity ? h : g + weight * h;

  // Initial state.
  const initReach = floodLocal(localOf, neighbours, startLocal, new Set(boxLocal0));
  let initCanon = n;
  for (let i = 0; i < n; i++) if (initReach[i] && i < initCanon) initCanon = i;
  const h0 = heuristic(boxLocal0);
  if (h0 >= INF || deadlocked(boxLocal0)) return null;
  addState(encode(boxLocal0, initCanon), 0, fOf(0, h0), -1, -1, -1, startLocal);

  let expanded = 0;
  while (heap.length > 0 && expanded < cap) {
    const s = heapPop();
    if (best.get(stKey[s]!) !== s) continue;
    const g = stG[s]!;
    const boxes = decodeBoxes(stKey[s]!);
    const playerEnd = stPlayerEnd[s]!;
    if (solved(boxes)) {
      if (!p.requireExit) return finish(s);
      const reach = floodLocal(localOf, neighbours, playerEnd, new Set(boxes));
      if (doorLocal.some((d) => reach[d])) return finish(s);
    }
    expanded++;
    const boxSet = new Set(boxes);
    const reach = floodLocal(localOf, neighbours, playerEnd, boxSet);
    for (let bi = 0; bi < m; bi++) {
      const b = boxes[bi]!;
      for (let di = 0; di < 4; di++) {
        const dest = neighbours[b]![di]!;
        if (dest < 0 || !isRoom[dest] || boxSet.has(dest)) continue;
        const from = neighbours[b]![di ^ 1]!;
        if (from < 0 || boxSet.has(from) || !reach[from]) continue;
        const nb = [...boxes];
        nb[bi] = dest;
        nb.sort((a, b2) => a - b2);
        if (deadlocked(nb)) continue;
        const h = heuristic(nb);
        if (h >= INF) continue;
        const reach2 = floodLocal(localOf, neighbours, b, new Set(nb));
        let canon = n;
        for (let i = 0; i < n; i++) if (reach2[i] && i < canon) canon = i;
        const key = encode(nb, canon);
        const g2 = g + 1;
        const prev = best.get(key);
        if (prev !== undefined && stG[prev]! <= g2) continue;
        addState(key, g2, fOf(g2, h), s, b, di, b);
      }
    }
  }
  return null;

  function finish(goalIdx: number): PushSolution {
    // Reconstruct the pushes, then turn each into walk + push moves.
    const pushes: { box: number; dir: number }[] = [];
    for (let t = goalIdx; t > 0; t = stParent[t]!) {
      pushes.push({ box: stPushBox[t]!, dir: stPushDir[t]! });
    }
    pushes.reverse();
    let moves = '';
    let curPlayer = startLocal;
    let curBoxes = [...boxLocal0];
    for (const push of pushes) {
      const from = neighbours[push.box]![push.dir ^ 1]!;
      const dest = neighbours[push.box]![push.dir]!;
      // Walk the player to `from` with the current boxes as obstacles.
      const walk = walkLocal(curPlayer, from, new Set(curBoxes));
      if (walk === null) return { moves: '', finalPlayer: -1, finalBoxes: [] };
      moves += walk;
      moves += DIRS[push.dir]!.ch.toUpperCase();
      curBoxes = curBoxes.map((b) => (b === push.box ? dest : b));
      curBoxes.sort((a, b) => a - b);
      curPlayer = push.box;
    }
    return {
      moves,
      finalPlayer: p.moveCells[curPlayer]!,
      finalBoxes: curBoxes.map((b) => p.moveCells[b]!),
    };
  }

  function walkLocal(from: number, to: number, blocked: Set<number>): string | null {
    if (from === to) return '';
    const prev = new Int32Array(n).fill(-1);
    const prevCh = new Array<string>(n).fill('');
    const q: number[] = [from];
    prev[from] = from;
    let head = 0;
    while (head < q.length) {
      const c = q[head++]!;
      for (let di = 0; di < 4; di++) {
        const m = neighbours[c]![di]!;
        if (m < 0 || prev[m] !== -1 || blocked.has(m)) continue;
        prev[m] = c;
        prevCh[m] = DIRS[di]!.ch;
        if (m === to) {
          let s = '';
          let cur = m;
          while (cur !== from) {
            s = prevCh[cur]! + s;
            cur = prev[cur]!;
          }
          return s;
        }
        q.push(m);
      }
    }
    return null;
  }
}

// Solve one room, trying optimal A* first, then weighted, then greedy.
function solveRoom(
  lv: Level,
  room: Room,
  entryCell: number,
  requireExit: boolean,
): PushSolution | null {
  const prob: PushProblem & { width: number } = {
    moveCells: [...room.cells, ...room.entries],
    roomCells: room.cells,
    goals: room.goals,
    boxes: room.boxes,
    startCell: entryCell,
    doors: room.doors,
    requireExit,
    width: lv.w,
  };
  return (
    solvePush(prob, 1, 250_000) ??
    solvePush(prob, 2, 250_000) ??
    solvePush(prob, Infinity, 250_000)
  );
}

// Whole-level fallback when the level does not split into rooms.
function solveWhole(lv: Level): string {
  const floor: number[] = [];
  for (let i = 0; i < lv.w * lv.h; i++) if (!lv.wall[i]) floor.push(i);
  const prob: PushProblem & { width: number } = {
    moveCells: floor,
    roomCells: floor,
    goals: lv.goals,
    boxes: lv.boxes,
    startCell: lv.player,
    doors: [],
    requireExit: false,
    width: lv.w,
  };
  const sol =
    solvePush(prob, 1, 200_000) ??
    solvePush(prob, 2, 200_000) ??
    solvePush(prob, Infinity, 200_000);
  return sol ? sol.moves : '';
}

function solveLevel(text: string): string {
  const lv = parseLevel(text);
  const rooms = decompose(lv);
  if (rooms === null) return solveWhole(lv);
  const work = rooms.filter((r) => r.boxes.length > 0);
  if (work.length === 0) return '';

  // Solve every room independently. Rooms that cannot be exited are solved
  // without the exit requirement and done last.
  type Done = { room: Room; sol: PushSolution; needsLast: boolean };
  const done: Done[] = [];
  for (const room of work) {
    const entry = room.entries[0]!;
    let sol = solveRoom(lv, room, entry, true);
    let needsLast = false;
    if (sol === null) {
      sol = solveRoom(lv, room, entry, false);
      needsLast = true;
    }
    if (sol === null || sol.moves === '' || sol.finalPlayer < 0) return '';
    done.push({ room, sol, needsLast });
  }
  const ordered = [
    ...done.filter((d) => !d.needsLast),
    ...done.filter((d) => d.needsLast),
  ];

  // Stitch: walk between rooms along the hallway, solve, walk out.
  let moves = '';
  let playerPos = lv.player;
  let curBoxes = [...lv.boxes];
  const last = ordered.length - 1;
  for (let ri = 0; ri < ordered.length; ri++) {
    const { room, sol } = ordered[ri]!;
    const entry = room.entries[0]!;
    const blocked = new Set(curBoxes);
    const walkIn = walkPath(lv, playerPos, entry, blocked);
    if (walkIn === null) return '';
    moves += walkIn;
    moves += sol.moves;
    // Update the box positions to this room's solved state.
    const roomBoxSet = new Set(room.boxes);
    curBoxes = [
      ...curBoxes.filter((b) => !roomBoxSet.has(b)),
      ...sol.finalBoxes,
    ];
    playerPos = sol.finalPlayer;
    if (ri !== last) {
      // Walk out through the door to the hallway.
      const blockedOut = new Set(curBoxes);
      let bestOut: { walk: string; entryCell: number } | null = null;
      for (const door of room.doors) {
        const w = walkPath(lv, playerPos, door, blockedOut);
        if (w === null) continue;
        // Step from the door to its hallway neighbour.
        for (const d of DIRS) {
          const nx = (door % lv.w) + d.dx;
          const ny = Math.floor(door / lv.w) + d.dy;
          if (nx < 0 || nx >= lv.w || ny < 0 || ny >= lv.h) continue;
          const m = ny * lv.w + nx;
          if (room.entries.includes(m) && !blockedOut.has(m)) {
            const cand = { walk: w + d.ch, entryCell: m };
            if (bestOut === null || cand.walk.length < bestOut.walk.length) {
              bestOut = cand;
            }
          }
        }
      }
      if (bestOut === null) return '';
      moves += bestOut.walk;
      playerPos = bestOut.entryCell;
    }
  }
  return moves;
}

for (const entry of readdirSync(INPUT_DIR, { withFileTypes: true }).sort((a, b) =>
  a.name.localeCompare(b.name),
)) {
  if (!entry.isDirectory()) continue;
  const id = entry.name;
  const moves = solveLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
