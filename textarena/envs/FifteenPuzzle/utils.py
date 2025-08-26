import random

def find_blank(board):
    for i in range(4):
        for j in range(4):
            if board[i][j] == 0:
                return i, j

def available_moves(blank_pos):
    moves = []
    r, c = blank_pos
    if r > 0:
        moves.append((r - 1, c))
    if r < 3:
        moves.append((r + 1, c))
    if c > 0:
        moves.append((r, c - 1))
    if c < 3:
        moves.append((r, c + 1))
    return moves

def swap(board, pos1, pos2):
    new_board = [row[:] for row in board]
    r1, c1 = pos1
    r2, c2 = pos2
    new_board[r1][c1], new_board[r2][c2] = new_board[r2][c2], new_board[r1][c1]
    return new_board

def generate_puzzle(swaps):
    for n_swaps in range(1,swaps+1):
        tiles = list(range(1, 16)) + [None]
        solved_board = [tiles[i:i + 4] for i in range(0, 16, 4)]
        board = [row[:] for row in solved_board]
        blank_pos = find_blank(board)
        last_blank = None
        for _ in range(n_swaps):
            moves = available_moves(blank_pos)
            if last_blank and last_blank in moves and len(moves) > 1:
                moves.remove(last_blank)
            next_pos = random.choice(moves)
            board = swap(board, blank_pos, next_pos)
            last_blank = blank_pos
            blank_pos = next_pos
    return board