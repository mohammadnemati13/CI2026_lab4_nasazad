# Lab 4: Generalized Klotski with BFS and A*

Solves the benchmark boards in `problems.txt` with Breadth-First Search and A*.

```
python lab4.py                 # all problems
python lab4.py --boards p01    # one problem, printing the board after every move
```

## Model
- **Move:** one piece slides one cell up, down, left or right. Every move costs 1.
- **Goal:** the target covers every bottom-row cell under the exit. Exit columns are 0-based: the exits are centred on every board, and p04 is the classic Klotski layout.
- **State:** the top-left cell of every piece. Non-target pieces with the same shape are interchangeable, so each shape group is stored sorted. This loses nothing, because the goal only looks at the target. It also keeps the search small: p02's ten 1×1 tiles alone would multiply the number of states by 10!.

## Algorithms
- **BFS:** FIFO frontier and a visited set; the goal test happens when a state is generated. It is optimal because every move costs 1.
- **A\*:** priority queue on f = g + h. Ties go to the lower h, and the goal test happens when a state is expanded.
  - h = Manhattan distance from the target to the goal + number of other pieces lying on the goal cells.
  - *Admissible:* the target needs at least that many moves of its own, and each blocking piece must move at least once.
  - *Consistent:* a single move changes only one of the two terms, by at most 1.

Every solution is replayed on a plain grid of piece names. This checks each move and the final position independently of the solver.

## Results
*Expanded* = states taken off the frontier. *States* = distinct states reached. Times are from one run on this machine.

| problem | size | optimal moves | BFS expanded | BFS states | BFS time | A* expanded | A* states | A* time | h(start) |
|---|---|---|---|---|---|---|---|---|---|
| p01 | 3×3 | 8 | 80 | 108 | 0.001 s | 33 | 65 | 0.001 s | 3 |
| p02 | 4×4 | 18 | 247 | 257 | 0.004 s | 174 | 211 | 0.004 s | 5 |
| p03 | 4×4 | 45 | 2,125 | 2,141 | 0.035 s | 2,013 | 2,065 | 0.037 s | 6 |
| p04 | 5×4 | 116 | 23,960 | 24,037 | 0.339 s | 23,637 | 23,770 | 0.424 s | 5 |
| p05 | 5×4 | 179 | 39,515 | 40,087 | 0.454 s | 36,607 | 37,441 | 0.825 s | 6 |
| p06 | 4×4 | no solution | 833 | 833 | 0.011 s | 833 | 833 | 0.013 s | 4 |

- **BFS and A\* always find the same optimal length.** p04 is the classic layout, and its 116 one-cell moves match the published optimum.
- **A\* expands fewer states on every solvable board, but the saving is large only on the small ones.** On p01 it expands 33 states against 80. On the hard boards the heuristic says 5–6 when the real cost is 45–179, because most of the work is moving other pieces out of the way. On p04 and p05 the priority queue and the heuristic cost more time than the few states they save, so A* is slower than BFS.
- **p06 has no solution.** Both searches explore all 833 reachable states without reaching the goal. With A at the bottom centre, rows 2–3 have no room for the horizontal pieces C and D, so both would have to be in rows 0–1. No reachable state has them both there.
