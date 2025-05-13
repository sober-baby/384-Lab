"""
An AI player for Othello. 
"""

import random
import sys
import time

from othello_shared import find_lines, get_possible_moves, get_score, play_move

cache = {}  # For state caching in minimax/alphabeta
possible_moves_cache = {}  # For memoizing get_possible_moves

def opposite(color):
    return 2 if color == 1 else 1

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
def board_to_key(board):
    return tuple(tuple(row) for row in board)

def memoized_get_possible_moves(board, color):
    key = (board_to_key(board), color)
    if key in possible_moves_cache:
        return possible_moves_cache[key]
    moves = get_possible_moves(board, color)
    possible_moves_cache[key] = moves
    return moves

def compute_utility(board, color):
    scores = get_score(board)
    return scores[0] - scores[1] if color == 1 else scores[1] - scores[0]

def compute_heuristic(board, color):
    n = len(board)
    corners = {(0,0), (0, n-1), (n-1, 0), (n-1, n-1)}
    dark, light, empty = 0, 0, 0
    my_corners = opp_corners = 0
    opp_color = opposite(color)

    for r in range(n):
        for c in range(n):
            val = board[r][c]
            if val == 0:
                empty +=1
            elif val == 1:
                dark +=1
            else:
                light +=1
            if (r,c) in corners:
                if val == color:
                    my_corners +=1
                elif val == opp_color:
                    opp_corners +=1

    my_score = dark if color ==1 else light
    opp_score = light if color ==1 else dark
    disk_diff = my_score - opp_score

    total = n*n
    filled = total - empty
    fraction = filled / total

    my_moves = len(memoized_get_possible_moves(board, color))
    opp_moves = len(memoized_get_possible_moves(board, opp_color))
    move_diff = my_moves - opp_moves

    corner_weight = 25
    corner_diff = (my_corners - opp_corners) * corner_weight

    if fraction < 0.33:
        return 3 * move_diff + corner_diff + disk_diff
    elif fraction < 0.8:
        return move_diff + 2 * (corner_diff + disk_diff)
    else:
        return move_diff + corner_diff + 4 * disk_diff

def minimax_min_node(board, color, limit, caching=0):
    opp = opposite(color)
    moves = memoized_get_possible_moves(board, opp)
    if not moves:
        return (None, compute_utility(board, color))
    if limit ==0:
        return (None, compute_heuristic(board, color))
    
    key = (board_to_key(board), color)
    if caching and key in cache:
        return cache[key]
    
    best_val = float('inf')
    best_move = None
    for move in moves:
        new_board = play_move(board, opp, *move)
        _, val = minimax_max_node(new_board, color, limit-1, caching)
        if val < best_val:
            best_val, best_move = val, move
    
    if caching:
        cache[key] = (best_move, best_val)
    return (best_move, best_val)

def minimax_max_node(board, color, limit, caching=0):
    moves = memoized_get_possible_moves(board, color)
    if not moves:
        return (None, compute_utility(board, color))
    if limit ==0:
        return (None, compute_heuristic(board, color))
    
    key = (board_to_key(board), color)
    if caching and key in cache:
        return cache[key]
    
    best_val = float('-inf')
    best_move = None
    for move in moves:
        new_board = play_move(board, color, *move)
        _, val = minimax_min_node(new_board, color, limit-1, caching)
        if val > best_val:
            best_val, best_move = val, move
    
    if caching:
        cache[key] = (best_move, best_val)
    return (best_move, best_val)

def select_move_minimax(board, color, limit, caching=0):
    cache.clear()
    possible_moves_cache.clear()
    best_move, _ = minimax_max_node(board, color, limit, caching)
    return best_move

def alphabeta_min_node(board, color, alpha, beta, limit, caching=0, ordering=0):
    opp = opposite(color)
    moves = memoized_get_possible_moves(board, opp)
    if not moves:
        return (None, compute_utility(board, color))
    if limit ==0:
        return (None, compute_heuristic(board, color))
    
    key = (board_to_key(board), color)
    if caching and key in cache:
        return cache[key]
    
    best_val = float('inf')
    best_move = None

    if ordering:
        moves.sort(key=lambda m: compute_heuristic(play_move(board, opp, *m), color), reverse=False)
    
    for move in moves:
        new_board = play_move(board, opp, *move)
        _, val = alphabeta_max_node(new_board, color, alpha, beta, limit-1, caching, ordering)
        if val < best_val:
            best_val, best_move = val, move
        if val < beta:
            beta = val
        if alpha >= beta:
            break
    
    if caching:
        cache[key] = (best_move, best_val)
    return (best_move, best_val)

def alphabeta_max_node(board, color, alpha, beta, limit, caching=0, ordering=0):
    moves = memoized_get_possible_moves(board, color)
    if not moves:
        return (None, compute_utility(board, color))
    if limit ==0:
        return (None, compute_heuristic(board, color))
    
    key = (board_to_key(board), color)
    if caching and key in cache:
        return cache[key]
    
    best_val = float('-inf')
    best_move = None

    if ordering:
        moves.sort(key=lambda m: compute_heuristic(play_move(board, color, *m), color), reverse=True)
    
    for move in moves:
        new_board = play_move(board, color, *move)
        _, val = alphabeta_min_node(new_board, color, alpha, beta, limit-1, caching, ordering)
        if val > best_val:
            best_val, best_move = val, move
        if val > alpha:
            alpha = val
        if alpha >= beta:
            break
    
    if caching:
        cache[key] = (best_move, best_val)
    return (best_move, best_val)

def select_move_alphabeta(board, color, limit=-1, caching=0, ordering=0):
    cache.clear()
    possible_moves_cache.clear()
    best_move, _ = alphabeta_max_node(board, color, -float('inf'), float('inf'), limit, caching, ordering)
    return best_move

# The rest of the run_ai() function remains unchanged.

####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state until the game is over.
    """
    print("Othello AI") # First line is the name of this AI
    arguments = input().split(",")
    
    color = int(arguments[0]) # Player color: 1 for dark (goes first), 2 for light. 
    limit = int(arguments[1]) # Depth limit
    minimax = int(arguments[2]) # Minimax or alpha beta
    caching = int(arguments[3]) # Caching 
    ordering = int(arguments[4]) # Node-ordering (for alpha-beta only)

    if (minimax == 1): eprint("Running MINIMAX")
    else: eprint("Running ALPHA-BETA")

    if (caching == 1): eprint("State Caching is ON")
    else: eprint("State Caching is OFF")

    if (ordering == 1): eprint("Node Ordering is ON")
    else: eprint("Node Ordering is OFF")

    if (limit == -1): eprint("Depth Limit is OFF")
    else: eprint("Depth Limit is ", limit)

    if (minimax == 1 and ordering == 1): eprint("Node Ordering should have no impact on Minimax")

    while True: # This is the main loop
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        # The first number is the score for player 1 (dark), the second for player 2 (light)
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()
        dark_score = int(dark_score_s)
        light_score = int(light_score_s)

        if status == "FINAL": # Game is over.
            print
        else:
            board = eval(input()) # Read in the input and turn it into a Python
                                  # object. The format is a list of rows. The
                                  # squares in each row are represented by
                                  # 0 : empty square
                                  # 1 : dark disk (player 1)
                                  # 2 : light disk (player 2)

            # Select the move and send it to the manager
            if (minimax == 1): # run this if the minimax flag is given
                movei, movej = select_move_minimax(board, color, limit, caching)
            else: # else run alphabeta
                movei, movej = select_move_alphabeta(board, color, limit, caching, ordering)
            
            print("{} {}".format(movei, movej))

if __name__ == "__main__":
    run_ai()
