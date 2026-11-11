import heapq
import sys
import textwrap
import time
from collections import deque
from itertools import count
from pathlib import Path

DIRECTIONS = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}


def parse_problems(path):
    problems, current, board = [], None, None
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if set(line) == {'='}:
            if board is None:
                board = []
            else:
                current['board'], board = board, None
        elif board is not None:
            board.append(line.split())
        else:
            key, *args = line.split()
            if key == 'problem':
                current = {'id': args[0]}
                problems.append(current)
            elif key == 'size':
                current['rows'], current['cols'] = map(int, args)
            elif key == 'exit' and args[0] == 'bottom':
                current['exit'] = [int(c) for c in args[1:]]
            elif key == 'target':
                current['target'] = args[0]
            else:
                raise ValueError(f'unsupported line: {line!r}')

    for p in problems:
        rows, cols = p['rows'], p['cols']
        if len(p['board']) != rows or any(len(row) != cols for row in p['board']):
            raise ValueError(f"{p['id']}: the board is not {rows}x{cols}")
        if not p['exit'] or not all(0 <= c < cols for c in p['exit']):
            raise ValueError(f"{p['id']}: exit columns out of range")
        if not any(p['target'] in row for row in p['board']):
            raise ValueError(f"{p['id']}: target {p['target']} is not on the board")
    return problems


class Klotski:
    def __init__(self, problem):
        self.rows, self.cols = rows, cols = problem['rows'], problem['cols']
        target = problem['target']

        cells = {}
        for r, row in enumerate(problem['board']):
            for c, token in enumerate(row):
                if token != '.':
                    cells.setdefault(token, []).append((r, c))
        pieces = {}
        for name, where in cells.items():
            top, left = min(r for r, _ in where), min(c for _, c in where)
            h = max(r for r, _ in where) - top + 1
            w = max(c for _, c in where) - left + 1
            if h * w != len(where):
                raise ValueError(f"{problem['id']}: piece {name} is not a rectangle")
            pieces[name] = (top, left, h, w)

        others = {}
        for name, (top, left, h, w) in pieces.items():
            if name != target:
                others.setdefault((h, w), []).append(top * cols + left)
        top, left, h, w = pieces[target]
        start = [top * cols + left]
        self.shapes = [(h, w)]
        self.group = [(0, 1)]
        for shape in sorted(others):
            begin = len(start)
            start += sorted(others[shape])
            self.shapes += [shape] * len(others[shape])
            self.group += [(begin, len(start))] * len(others[shape])
        self.start = tuple(start)

        self.mask, self.step = {}, {}
        for shape in set(self.shapes):
            h, w = shape
            self.mask[shape], self.step[shape] = {}, {}
            for r in range(rows - h + 1):
                for c in range(cols - w + 1):
                    self.mask[shape][r * cols + c] = sum(
                        1 << ((r + i) * cols + c + j) for i in range(h) for j in range(w))
                    self.step[shape][r * cols + c] = {
                        d: (r + dr) * cols + c + dc
                        for d, (dr, dc) in DIRECTIONS.items()
                        if 0 <= r + dr <= rows - h and 0 <= c + dc <= cols - w}

        h, w = self.shapes[0]
        exits = problem['exit']
        self.goals = {(rows - h) * cols + c
                      for c in range(max(0, max(exits) - w + 1), min(min(exits), cols - w) + 1)}
        if not self.goals:
            raise ValueError(f"{problem['id']}: the target cannot cover all the exit cells")
        self.goal_cells = [divmod(g, cols) for g in self.goals]
        self.must_clear = -1
        for g in self.goals:
            self.must_clear &= self.mask[self.shapes[0]][g]

    def successors(self, state):
        mask = self.mask
        occupied = 0
        for shape, pos in zip(self.shapes, state):
            occupied |= mask[shape][pos]
        for i, (shape, pos) in enumerate(zip(self.shapes, state)):
            rest = occupied ^ mask[shape][pos]
            for d, new in self.step[shape][pos].items():
                if not (mask[shape][new] & rest):
                    nxt = list(state)
                    nxt[i] = new
                    begin, end = self.group[i]
                    if end - begin > 1:
                        nxt[begin:end] = sorted(nxt[begin:end])
                    yield tuple(nxt), (pos, d)

    def is_goal(self, state):
        return state[0] in self.goals

    def heuristic(self, state):
        r, c = divmod(state[0], self.cols)
        distance = min(abs(r - gr) + abs(c - gc) for gr, gc in self.goal_cells)
        blockers = sum(1 for shape, pos in zip(self.shapes[1:], state[1:])
                       if self.mask[shape][pos] & self.must_clear)
        return distance + blockers


def bfs(puzzle):
    started = time.perf_counter()
    parent = {puzzle.start: None}
    frontier = deque([puzzle.start])
    expanded = 0
    goal = puzzle.start if puzzle.is_goal(puzzle.start) else None
    while frontier and goal is None:
        state = frontier.popleft()
        expanded += 1
        for nxt, move in puzzle.successors(state):
            if nxt not in parent:
                parent[nxt] = (state, move)
                if puzzle.is_goal(nxt):
                    goal = nxt
                    break
                frontier.append(nxt)
    return search_result(parent, goal, expanded, started)


def astar(puzzle):
    started = time.perf_counter()
    h = puzzle.heuristic(puzzle.start)
    tie = count()
    frontier = [(h, h, next(tie), 0, puzzle.start)]
    best_g = {puzzle.start: 0}
    parent = {puzzle.start: None}
    expanded, goal = 0, None
    while frontier:
        _, _, _, g, state = heapq.heappop(frontier)
        if g > best_g[state]:
            continue
        expanded += 1
        if puzzle.is_goal(state):
            goal = state
            break
        for nxt, move in puzzle.successors(state):
            if g + 1 < best_g.get(nxt, float('inf')):
                best_g[nxt] = g + 1
                parent[nxt] = (state, move)
                h = puzzle.heuristic(nxt)
                heapq.heappush(frontier, (g + 1 + h, h, next(tie), g + 1, nxt))
    return search_result(parent, goal, expanded, started)


def search_result(parent, goal, expanded, started):
    moves = None
    if goal is not None:
        moves = []
        while parent[goal] is not None:
            goal, move = parent[goal]
            moves.append(move)
        moves.reverse()
    return {'moves': moves, 'expanded': expanded, 'states': len(parent),
            'seconds': time.perf_counter() - started}


def replay(problem, moves):
    rows, cols = problem['rows'], problem['cols']
    grid = [row[:] for row in problem['board']]
    named, boards = [], []
    for pos, d in moves:
        r, c = divmod(pos, cols)
        name = grid[r][c]
        cells = [(i, j) for i in range(rows) for j in range(cols) if grid[i][j] == name]
        assert name != '.' and min(cells) == (r, c), f'no piece has its top-left cell at {r, c}'
        dr, dc = DIRECTIONS[d]
        for i, j in cells:
            grid[i][j] = '.'
        for i, j in cells:
            assert 0 <= i + dr < rows and 0 <= j + dc < cols and grid[i + dr][j + dc] == '.', \
                f'illegal move: {name} {d}'
            grid[i + dr][j + dc] = name
        named.append(f'{name} {d}')
        boards.append([row[:] for row in grid])
    assert all(grid[rows - 1][c] == problem['target'] for c in problem['exit']), \
        'the target does not cover the exit'
    return named, boards


def show(grid, indent='    '):
    width = max(len(token) for row in grid for token in row)
    return '\n'.join(indent + ' '.join(token.rjust(width) for token in row) for row in grid)


def main():
    show_boards = '--boards' in sys.argv
    args = [a for a in sys.argv[1:] if a != '--boards']
    files = [a for a in args if a.endswith('.txt')]
    only = set(args) - set(files)
    path = Path(files[0]) if files else Path(__file__).with_name('problems.txt')

    summary = []
    for problem in parse_problems(path):
        if only and problem['id'] not in only:
            continue
        puzzle = Klotski(problem)
        print(f"=== {problem['id']}: {problem['rows']}x{problem['cols']} board, "
              f"target {problem['target']}, exit under columns {problem['exit']} (0-based)")
        print(show(problem['board']))
        runs = {'BFS': bfs(puzzle), 'A*': astar(puzzle)}
        for name, run in runs.items():
            if run['moves'] is None:
                print(f"  {name}: no solution (all {run['states']} reachable states explored)")
                continue
            run['named'], run['boards'] = replay(problem, run['moves'])
            print(textwrap.fill(f"{name} ({len(run['moves'])} moves): " + ', '.join(run['named']),
                                width=100, initial_indent='  ', subsequent_indent='      '))
        lengths = {None if run['moves'] is None else len(run['moves']) for run in runs.values()}
        assert len(lengths) == 1, 'BFS and A* disagree on the optimal number of moves'

        best = runs['A*']
        if best['moves'] is not None and show_boards:
            for i, (move, board) in enumerate(zip(best['named'], best['boards']), 1):
                print(f'  {i}. {move}')
                print(show(board))
        elif best['moves']:
            print('  final board:')
            print(show(best['boards'][-1]))
        print()
        summary.append((problem, runs))

    print(f"{'problem':<8}{'size':>5}{'moves':>7}  |{'BFS expanded':>14}{'states':>9}{'time':>9}"
          f"  |{'A* expanded':>13}{'states':>9}{'time':>9}")
    for problem, runs in summary:
        moves = runs['BFS']['moves']
        line = (f"{problem['id']:<8}{str(problem['rows']) + 'x' + str(problem['cols']):>5}"
                f"{'-' if moves is None else len(moves):>7}")
        for name, width in (('BFS', 14), ('A*', 13)):
            run = runs[name]
            line += f"  |{run['expanded']:>{width}}{run['states']:>9}{run['seconds']:>8.3f}s"
        print(line)


if __name__ == '__main__':
    main()


# Peer review (Lab 4) by Mohammad Nemati:
# I ran the code on all six problems and checked the results.
# BFS and A* give the same optimal number of moves (8, 18, 45, 116, 179) and p06 has no solution.
# The heuristic is admissible: the target needs at least `distance` moves and every blocker
# on the goal cells needs at least one move. Bitmasks and sorted same-shape groups make it very fast.
