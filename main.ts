// roblox-up-and-coming-forecast-01 entry (Node 22, stdlib only).
// Model: a learned place-transition table + a blended recent-rank chart.
//  1. Read /input/history.ndjson (Up-and-Coming chart reads) and /input/task.json.
//  2. Train a transition table T[place][next] from consecutive history reads (with Laplace prior),
//     counting a drop below place 10 as "out" and letting each game spread mass over places 1..10.
//  3. Rank games by average position over the last 2 reads (top 18), weight in near-chart games too.
//  4. For each of the task's hourly slots, emit the same forecast: P(game at place j) from the
//     transition row of its blended rank, capped at 0.8 total named mass per place; the rest stays
//     unnamed so the scorer's unnamed share handles surprises. top10 = blended chart order.
//
// Outside the grader's container, MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR stand in for /input, /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Part 1 (this file continues in the model section below).
const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8'));
const lines = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8').split('\n').filter((l) => l !== '');
const reads = lines.map((l) => JSON.parse(l));
const PLACES = task.places; // 10
const BLEND_W = 2;          // reads blended for the current chart
const BLEND_DEPTH = 18;     // games considered per read
const PRIOR = 2;            // Laplace prior per (place,next) cell
const CAP = 0.8;            // max total named mass per place
const ENTRY_P = 0.03;       // per-place mass for near-chart games not in the top 10

// Part 2 — model. Transition table from consecutive reads: a game at place k lands at j (1..10) or out (11).
const T: Map<number, Map<number, number>> = new Map();
for (let i = 0; i + 1 < reads.length; i++) {
  const prev = new Map<number, number>();
  for (const row of reads[i]!.up_and_coming.slice(0, 15)) prev.set(row.universe_id, row.rank);
  const next = new Map<number, number>();
  for (const row of reads[i + 1]!.up_and_coming.slice(0, 15)) next.set(row.universe_id, row.rank);
  for (const [uid, k] of prev) {
    const j = next.get(uid) ?? 11;
    if (!T.has(k)) T.set(k, new Map());
    const rowT = T.get(k)!;
    rowT.set(Math.min(j, 11), (rowT.get(Math.min(j, 11)) ?? 0) + 1);
  }
}
function tp(k: number, j: number): number {
  const rowT = T.get(k) ?? new Map();
  let tot = PRIOR * 10;
  for (const v of rowT.values()) tot += v;
  return ((rowT.get(j) ?? 0) + PRIOR) / tot;
}

// Blended chart: average position over the last BLEND_W reads.
const pos = new Map<number, number>();
for (const rd of reads.slice(-BLEND_W)) {
  for (const row of rd.up_and_coming.slice(0, BLEND_DEPTH)) {
    pos.set(row.universe_id, (pos.get(row.universe_id) ?? 0) + row.rank);
  }
}
const nW = Math.min(BLEND_W, reads.length);
const ranked = [...pos.entries()]
  .sort((a, b) => a[1] / nW - b[1] / nW)
  .slice(0, BLEND_DEPTH)
  .map(([uid]) => uid);
const chart: number[] = ranked;
const top10 = new Set(chart.slice(0, PLACES));

// Forecast: top-10 games spread their transition row over places 1..10; games at blended ranks
// 11-20 (not in the top 10) get only small ENTRY_P rows at places 6-10 as entry chances.
type Prob = { universe_id: number; place: number; p: number };
const probs: Prob[] = [];
for (let k = 1; k <= PLACES && k <= chart.length; k++) {
  const u = chart[k - 1]!;
  for (let j = 1; j <= PLACES; j++) probs.push({ universe_id: u, place: j, p: tp(k, j) });
}
for (let k = PLACES + 1; k <= 20; k++) {
  const u = chart[k - 1];
  if (u === undefined || top10.has(u)) continue;
  for (let j = 6; j <= PLACES; j++) probs.push({ universe_id: u, place: j, p: ENTRY_P });
}
// Cap the named mass per place; leave the remainder unnamed for the scorer's share.
for (let j = 1; j <= PLACES; j++) {
  let tot = 0;
  for (const pr of probs) if (pr.place === j) tot += pr.p;
  if (tot > CAP) {
    const f = CAP / tot;
    for (const pr of probs) if (pr.place === j) pr.p *= f;
  }
}

// Output: ONE forecast object — {probabilities, top10} — scored against every hourly read of the
// 72-hour window. Constraint checks: each place sums to <= 1, a game at most once per place.
let sumByPlace = new Array(PLACES + 1).fill(0);
for (const pr of probs) sumByPlace[pr.place]! += pr.p;
for (let j = 1; j <= PLACES; j++) {
  if (sumByPlace[j]! > 1.000001) throw new Error(`place ${j} sums to ${sumByPlace[j]}`);
}
const seenAt = new Map<number, Set<number>>();
for (const pr of probs) {
  if (!seenAt.has(pr.universe_id)) seenAt.set(pr.universe_id, new Set());
  const s = seenAt.get(pr.universe_id)!;
  if (s.has(pr.place)) throw new Error(`game ${pr.universe_id} named twice at place ${pr.place}`);
  s.add(pr.place);
}
const outName = task.output ?? 'forecast.json';
writeFileSync(join(OUTPUT_DIR, outName), `${JSON.stringify({ probabilities: probs, top10: chart.slice(0, PLACES) })}\n`);
console.log(`forecast written: ${probs.length} probability rows, top10 ${chart.slice(0, PLACES).join(',')}`);
