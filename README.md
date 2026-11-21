# Sokoban solver

A push-optimal Sokoban solver written for make.fun's
"Build the best Sokoban solver, 21 November 2026" bounty.

## How it works

The generator builds every level from a one-wide hallway plus 1–4 rooms,
each room opening onto the hallway through a single door, with 3–4 boxes and
as many goals. A box pushed out of a room through its door can never come
back, so in an optimal solution every box stays in its room and a level's
push-optimal count is the sum of its rooms' optima. This program therefore:

1. splits each level into its hallway and its rooms (connected components of
   floor off the hallway row),
2. solves each room push-optimally on its own with A* over pushes — a state
   is the box positions plus the player's reachable region, the heuristic is
   the min-cost matching of boxes to goals, and successors are pruned with
   static dead squares, 2x2 blocks and frozen-box detection,
3. stitches the room solutions together with plain walking moves along the
   hallway.

Only the push count is scored, so walking moves are free. The search is
capped by expansions, never by the clock, so two runs write identical bytes.

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

`main.ts` is the whole entry: it reads `/input/<id>/level.txt` for every
instance and writes `/output/<id>/solution.txt` (moves in `u/d/l/r`, an
upper-case letter marking a push).
