"""
An AI player for Othello. 
"""

import random
import sys
import time

# You can use the functions from othello_shared to write your AI
from othello_shared import find_lines, get_possible_moves, get_score, play_move

cache = {} # Use this for state caching
def opposite(color):
    return 2 if color == 1 else 1

def eprint(*args, **kwargs): #use this for debugging, to print to sterr
    print(*args, file=sys.stderr, **kwargs)
    
def compute_utility(board, color):
    # IMPLEMENT!
    """
    Method to compute the utility value of board.
    INPUT: a game state and the player that is in control
    OUTPUT: an integer that represents utility
    """
    if color == 1:
        return get_score(board)[0] - get_score(board)[1]
    else:
        return get_score(board)[1] - get_score(board)[0]
    

def compute_heuristic(board, color):
    # IMPLEMENT! 
    """
    Method to heuristic value of board, to be used if we are at a depth limit.
    INPUT: a game state and the player that is in control
    OUTPUT: an integer that represents heuristic value
    """
    
    # setup
    n = len(board)
    def sgn(color): # black 1， white -1
        return (1.5 - color)*2
    
    def get_surrounding(i, j, n):
        directions = [(-1, -1), (-1, 0), (-1, 1),  # 左上、上、右上
                    (0, -1),          (0, 1),   # 左、     右
                    (1, -1),  (1, 0), (1, 1)]   # 左下、下、右下

        surrounding = []
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < n:  # 检查边界
                surrounding.append((ni, nj))
        return surrounding
    
    def get_surrounding_stra(i, j, n):
        directions = [       (-1, 0),           # 左上、上、右上
                    (0, -1),          (0, 1),   # 左、     右
                              (1, 0)        ]   # 左下、下、右下

        surrounding = []
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < n:  # 检查边界
                surrounding.append((ni, nj))
        return surrounding
    
    def get_surrounding_diag(i, j, n):
        directions = [(-1, -1),      (-1, 1),  # 左上、上、右上

                    (1, -1),            (1, 1)]   # 左下、下、右下

        surrounding = []
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < n:  # 检查边界
                surrounding.append((ni, nj))
        return surrounding
    
    # frontier: a piece who has a surrounding that is empty
    def isfrontier(board, i, j):
        if board[i][j] == 0: return False # double check
        sur = get_surrounding(i, j, len(board))
        for x, y in sur:
            if board[x][y] == 0: return True
        return False

    # corner control: play on corner, thus dont play on danger spots (neighbour of corner)
    corner = 0
    for i,j in [(0,0), (0,n-1), (n-1,0), (n-1,n-1)]:
        if board[i][j] == color:        corner += 15
        elif board[i][j] == 3 - color:  corner -= 15
        else:
            sur_stra = get_surrounding_stra(i, j, n)
            for x, y in sur_stra:
                if board[x][y] == color:        corner -= 3
                elif board[x][y] == 3 - color:  corner += 3
            
            sur_diag = get_surrounding_diag(i, j, n)
            for x, y in sur_diag:
                if board[x][y] == color:        corner -= 6
                elif board[x][y] == 3 - color:  corner += 6

    # 稳定子计算
    # 首先一个子必须在四个方向上均稳定才是稳定的
    # 由于子的反转只能由落子确定（而不能由落子引起的反转连锁），因此在一个方向上稳定要求
    # 在这个方向上连接这个子的一串棋，两侧均不存在空位，或一端是墙（故而己方稳定子）
    # 实际上，虽然存在理论上的两端都是敌方稳定子的安稳点，但可以推出敌方子靠墙故两侧无空位。
    # 因此一个子如果两侧是敌方稳定子，或者一侧是己方稳定子，则在这个方向安定
    # 稳定子一定是由角落开始不断延展，故

    def find_stability(board, color): #时间原因暂时不进行
        """
        计算当前玩家的稳定子数量（己方稳定子 - 对方稳定子）。
        :param board: 棋盘状态（二维列表）
        :param color: 当前玩家的颜色
        :return: 稳定子差值
        """
        n = len(board)
        opponent_color = 3 - color
        ct = 0  # 己方稳定子 - 对方稳定子
        directions = [
            (0, 1),   # 水平右
            (1, 0),   # 垂直下
            (1, 1),   # 主对角线右下
            (1, -1)   # 副对角线左下
        ]
        stable = [[0 for _ in range(n)] for _ in range(n)]  # 0: 未确定, 1: 己方稳定子, 2: 对方稳定子

        def is_stable(i, j):
            """
            检查位置 (i, j) 是否在四个方向上均稳定。
            """
            if board[i][j] == 0:
                return False
            current_color = board[i][j]
            for dx, dy in directions:
                stable_in_this_direction = False
                # 方向检查：向正反两个方向延伸
                for step in [1, -1]:
                    x, y = i, j
                    has_wall = False
                    has_same_stable = False
                    has_opponent_stable = False
                    while True:
                        x += dx * step
                        y += dy * step
                        if not (0 <= x < n and 0 <= y < n):
                            has_wall = True
                            break
                        if board[x][y] == 0:
                            break  # 遇到空位，方向不稳定
                        if stable[x][y] == current_color:
                            has_same_stable = True
                            break
                        elif stable[x][y] == 3 - current_color:
                            has_opponent_stable = True
                    # 判断方向是否稳定
                    if has_wall or has_same_stable:
                        stable_in_this_direction = True
                    elif has_opponent_stable:
                        # 两侧均为对方稳定子
                        x2, y2 = i - dx * step, j - dy * step
                        if 0 <= x2 < n and 0 <= y2 < n and stable[x2][y2] == 3 - current_color:
                            stable_in_this_direction = True
                    if not stable_in_this_direction:
                        return False
            return True

        # 逐层检测：从最外层到中心
        for layer in range((n + 1) // 2):
            # 当前层的四个角
            corners = [
                (layer, layer),
                (layer, n - 1 - layer),
                (n - 1 - layer, layer),
                (n - 1 - layer, n - 1 - layer)
            ]
            # 检查四个角
            for i, j in corners:
                if board[i][j] != 0 and stable[i][j] == 0 and is_stable(i, j):
                    stable_color = 1 if board[i][j] == color else 2
                    stable[i][j] = stable_color
                    ct += 1 if stable_color == 1 else -1

            # 从四个角沿水平和垂直方向延伸
            for i, j in corners:
                # 水平延伸（左右）
                for dj in [-1, 1]:
                    y = j + dj
                    while layer <= y <= n - 1 - layer and layer <= i <= n - 1 - layer:
                        if board[i][y] != 0 and stable[i][y] == 0 and is_stable(i, y):
                            stable_color = 1 if board[i][y] == color else 2
                            stable[i][y] = stable_color
                            ct += 1 if stable_color == 1 else -1
                        y += dj
                # 垂直延伸（上下）
                for di in [-1, 1]:
                    x = i + di
                    while layer <= x <= n - 1 - layer and layer <= j <= n - 1 - layer:
                        if board[x][j] != 0 and stable[x][j] == 0 and is_stable(x, j):
                            stable_color = 1 if board[x][j] == color else 2
                            stable[x][j] = stable_color
                            ct += 1 if stable_color == 1 else -1
                        x += di

        return ct    
    def find_stability_easy(board, color):

        n = len(board)
        opponent_color = 3 - color
        ct = 0  # 己方稳定子 - 对方稳定子
        visited = set()  # 用于标记已经处理过的棋子

        # 1. 检查四条边是否满
        def is_edge_full(edge):
            return all(cell != 0 for cell in edge)

        top_edge = board[0]
        if is_edge_full(top_edge):
            for j in range(n):
                if (0, j) not in visited:
                    if board[0][j] == color:
                        ct += 1
                    elif board[0][j] == opponent_color:
                        ct -= 1
                    visited.add((0, j))

        bottom_edge = board[n-1]
        if is_edge_full(bottom_edge):
            for j in range(n):
                if (n-1, j) not in visited:
                    if board[n-1][j] == color:
                        ct += 1
                    elif board[n-1][j] == opponent_color:
                        ct -= 1
                    visited.add((n-1, j))

        left_edge = [board[i][0] for i in range(n)]
        if is_edge_full(left_edge):
            for i in range(n):
                if (i, 0) not in visited:
                    if board[i][0] == color:
                        ct += 1
                    elif board[i][0] == opponent_color:
                        ct -= 1
                    visited.add((i, 0))

        right_edge = [board[i][n-1] for i in range(n)]
        if is_edge_full(right_edge):
            for i in range(n):
                if (i, n-1) not in visited:
                    if board[i][n-1] == color:
                        ct += 1
                    elif board[i][n-1] == opponent_color:
                        ct -= 1
                    visited.add((i, n-1))

        # 2. 检查四个角
        corners = [(0, 0), (0, n-1), (n-1, 0), (n-1, n-1)]
        for i, j in corners:
            if board[i][j] != 0 and (i, j) not in visited:
                current_color = board[i][j]

                for dj in [-1, 1]:
                    y = j + dj
                    while 0 <= y < n and board[i][y] == current_color and (i, y) not in visited:
                        if current_color == color:
                            ct += 1
                        else:
                            ct -= 1
                        visited.add((i, y))
                        y += dj

                for di in [-1, 1]:
                    x = i + di
                    while 0 <= x < n and board[x][j] == current_color and (x, j) not in visited:
                        if current_color == color:
                            ct += 1
                        else:
                            ct -= 1
                        visited.add((x, j))
                        x += di

        return ct 
    stability = find_stability_easy(board, color)

    # current difference
    dark, light = get_score(board) # existing function
    current = (dark - light)*sgn(color)
    
    # fragile pieces (frontier)
    frontier = 0
    for i in range(n):
        for j in range(n):
            c = board[i][j]
            if c == 0: continue
            if isfrontier(board, i, j):
                if c == color:  frontier -= 1
                else:           frontier += 1


    # 以下的思路在于：限制对方落子

    # Mobility calculation
    # 由于本函数仅能判断局面因此没法判断一步棋对棋局的影响，但是可以判断行动力差
    ct_mvs_self = len(get_possible_moves(board, color))
    ct_mvs_oppo = len(get_possible_moves(board, 3 - color))
    mobility = ct_mvs_self - ct_mvs_oppo

    # 计算奇偶性
    # 由于是非标规，也即无棋可走就结束游戏，奇偶性的效果有待验证; 事件原因暂时不进行
    parity = 0

    # if end then find result
    if ct_mvs_self == 0 or ct_mvs_oppo == 0: 
        return current*float('inf')

    # corner自带weight：
    # 角 = 90 ； 危险点 = -18 / -36
    return 6*corner + 11*stability + 0.5*current + 2*frontier + 2*mobility + parity

# ...okay so playmove must pass in i, j not (i,j)..........................~!!~!!!!!!~!!@@~!@~!@~!@~!@~!


############ MINIMAX ###############################
def minimax_min_node(board, color, limit, caching = 0):
    # IMPLEMENT!
    """
    A helper function for minimax that finds the lowest possible utility
    """
    # HINT:
    # 1. Get the allowed moves
    # 2. Check if w are at terminal state
    # 3. If not, for each possible move, get the max utiltiy
    # 4. After checking every move, you can find the minimum utility
    # ...
    
    opp = opposite(color)
        
    moves = get_possible_moves(board, opp)
    
    if not moves:
        return (None, compute_utility(board, color))
    if limit == 0:
        return (None, compute_heuristic(board, color))
    
    #check cashing
    
    cache_key = ("MIN", board, color, limit)
    if caching and cache_key in cache:
        return cache[cache_key]
    
    best_value = float('inf')
    best_move = None
    
    for move in moves:
        new_board = play_move(board, opp, move[0], move[1])
        _, val = minimax_max_node(new_board, color, limit - 1, caching)
        if val < best_value:
            best_value = val
            best_move = move
    
    
    if caching:
        cache[cache_key] = (best_move, best_value)
        
    return (best_move, best_value)
    


def minimax_max_node(board, color, limit, caching = 0):
    # IMPLEMENT!
    """
    A helper function for minimax that finds the highest possible utility
    """
    # HINT:
    # 1. Get the allowed moves
    # 2. Check if w are at terminal state
    # 3. If not, for each possible move, get the min utiltiy
    # 4. After checking every move, you can find the maximum utility
    # ...
    moves = get_possible_moves(board, color)
    
    if not moves:
        return (None, compute_utility(board, color))
    if limit == 0:
        return (None, compute_heuristic(board, color))
    
    #check cashing
    
    cache_key = ("MAX", board, color, limit)
    if caching and cache_key in cache:
        return cache[cache_key]
    
    best_value = float('-inf')
    best_move = None
    
    for move in moves:
        new_board = play_move(board, color, move[0], move[1])
        _, val = minimax_min_node(new_board, color, limit - 1, caching)
        if val > best_value:
            best_value = val
            best_move = move
    
    if caching:
        cache[cache_key] = (best_move, best_value)
        
    return (best_move, best_value)

    
def select_move_minimax(board, color, limit, caching = 0):
    # IMPLEMENT!
    """
    Given a board and a player color, decide on a move using Minimax algorithm. 
    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic 
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.
    INPUT: a game state, the player that is in control, the depth limit for the search, and a flag determining whether state caching is on or not
    OUTPUT: a tuple of integers (i,j) representing a move, where i is the column and j is the row on the board.
    """
    best_move, _ = minimax_max_node(board, color, limit, caching)
    return best_move


############ ALPHA-BETA PRUNING #####################
def alphabeta_min_node(board, color, alpha, beta, limit, caching = 0, ordering = 0):
    
    opp = opposite(color)
    moves = get_possible_moves(board, opp)
    if not moves:
        return (None, compute_utility(board, color))
    if limit == 0:
        return (None, compute_heuristic(board, color))
    
    #check cashing
    cache_key = ("AB_MIN", board, color, limit, alpha, beta)
    if caching and cache_key in cache:
        return cache[cache_key]
    
    best_value = float('inf')
    best_move = None
    
    #ordering moves based on highest heuristic value
    if ordering:
        moves = sorted(moves, key = lambda x: compute_heuristic(play_move(board, opp, x[0], x[1]), color), reverse=True)
    
    for move in moves:
        new_board = play_move(board, opp, move[0], move[1])
        _, val = alphabeta_max_node(new_board, color, alpha, beta, limit - 1, caching, ordering)
        if val < best_value:
            best_value = val
            best_move = move
        beta = min(beta, best_value)
        if beta <= alpha:
            break
        
    if caching:
        cache[cache_key] = (best_move, best_value)
    
    return (best_move, best_value)
        

def alphabeta_max_node(board, color, alpha, beta, limit, caching = 0, ordering = 0):
    # IMPLEMENT!

    moves = get_possible_moves(board, color)
    if not moves:
        return (None, compute_utility(board, color))
    if limit == 0:
        return (None, compute_heuristic(board, color))
    
    #check cashing
    cache_key = ("AB_MAX", board, color, limit, alpha, beta)
    if caching and cache_key in cache:
        return cache[cache_key]
    
    best_value = float('-inf')
    best_move = None
    
    if ordering:
        moves = sorted(moves, key = lambda x: compute_heuristic(play_move(board, color, x[0], x[1]), color), reverse=True)
        
    for move in moves:
        new_board = play_move(board, color, move[0], move[1])
        _, val = alphabeta_min_node(new_board, color, alpha, beta, limit - 1, caching, ordering)
        if val > best_value:
            best_value = val
            best_move = move
        alpha = max(alpha, best_value)
        if alpha >= beta:
            break
    
    if caching:
        cache[cache_key] = (best_move, best_value)
    
    return (best_move, best_value)

def select_move_alphabeta(board, color, limit = -1, caching = 0, ordering = 0):
    # IMPLEMENT!
    """
    Given a board and a player color, decide on a move using Alpha-Beta algorithm. 
    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic 
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.    
    If ordering is ON (i.e. 1), use node ordering to expedite pruning and reduce the number of state evaluations. 
    If ordering is OFF (i.e. 0), do NOT use node ordering to expedite pruning and reduce the number of state evaluations. 
    INPUT: a game state, the player that is in control, the depth limit for the search, a flag determining whether state caching is on or not, a flag determining whether node ordering is on or not
    OUTPUT: a tuple of integers (i,j) representing a move, where i is the column and j is the row on the board.
    """
    best_move, _ = alphabeta_max_node(board, color, float('-inf'), float('inf'), limit, caching, ordering)  
    return best_move

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
