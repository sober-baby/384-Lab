# Look for #IMPLEMENT tags in this file.
'''
All models need to return a CSP object, and a list of lists of Variable objects
representing the board. The returned list of lists is used to access the
solution.

For example, after these three lines of code

    csp, var_array = futoshiki_csp_model_1(board)
    solver = BT(csp)
    solver.bt_search(prop_FC, var_ord)

var_array[0][0].get_assigned_value() should be the correct value in the top left
cell of the Futoshiki puzzle.

1. futoshiki_csp_model_1 (worth 20/100 marks)
    - A model of a Futoshiki grid built using only
      binary not-equal constraints for both the row and column constraints.

2. futoshiki_csp_model_2 (worth 20/100 marks)
    - A model of a Futoshiki grid built using only n-ary
      all-different constraints for both the row and column constraints.
    
    The input board is specified as a list of n lists. Each of the n lists
    represents a row of the board. If a 0 is in the list it represents an empty
    cell. Otherwise if a number between 1--n is in the list then this
    represents a pre-set board position.

    Each list is of length 2n-1, with each space on the board being separated
    by the potential inequality constraints. '>' denotes that the previous
    space must be bigger than the next space; '<' denotes that the previous
    space must be smaller than the next; '.' denotes that there is no
    inequality constraint.

    E.g., the board

    -------
    | > |2|
    | | | |
    | | < |
    -------
    would be represented by the list of lists

    [[0,>,0,.,2],
     [0,.,0,.,0],
     [0,.,0,<,0]]

'''
import cspbase
import itertools

from cspbase import *

def futoshiki_csp_model_1(futo_grid):
    n = len(futo_grid)       # size of board
    
    # create all varibles
    var_array = []
    for i in range(n):
        rv = []
        for j in range(n):
            
            val = futo_grid[i][2*j]  
            if val == 0:
                domain = list(range(1, n+1))
            else:
                domain = [val]
                
            var = Variable(f"V({i},{j})", domain)
            rv.append(var)
        var_array.append(rv)

    # create csp object and add variables
    csp = CSP("Futoshiki_Model_1")
    for i in range(n):
        for j in range(n):
            csp.add_var(var_array[i][j])

    # add all row constraints
    for i in range(n):
        for j in range(n):
            for k in range(j+1, n):
                v1 = var_array[i][j]
                v2 = var_array[i][k]
                con = Constraint(f"RNEQ({i},{j},{k})", [v1, v2])
                sat_tuples = []
                
                for a in v1.domain():
                    for b in v2.domain():
                        if a != b:
                            sat_tuples.append((a, b))
                            
                con.add_satisfying_tuples(sat_tuples)
                csp.add_constraint(con)

        for j in range(n-1):
            sign = futo_grid[i][2*j + 1]
            
            if sign == '<':
                v_left  = var_array[i][j]
                v_right = var_array[i][j+1]
                con = Constraint(f"RL({i},{j})", [v_left, v_right])
                sat_tuples = []
                
                for a in v_left.domain():
                    for b in v_right.domain():
                        if a < b:
                            sat_tuples.append((a, b))
                            
                con.add_satisfying_tuples(sat_tuples)
                csp.add_constraint(con)
                
            elif sign == '>':
                v_left  = var_array[i][j]
                v_right = var_array[i][j+1]
                con = Constraint(f"RG({i},{j})", [v_left, v_right])
                sat_tuples = []
                
                for a in v_left.domain():
                    for b in v_right.domain():
                        if a > b:
                            sat_tuples.append((a, b))
                            
                con.add_satisfying_tuples(sat_tuples)
                csp.add_constraint(con)

    # add all coloumn constraints
    for i in range(n):
        for j in range(n):
            for k in range(j+1, n):
                v1 = var_array[j][i]
                v2 = var_array[k][i]
                con = Constraint(f"CNEQ({j},{j},{k})", [v1, v2])
                sat_tuples = []
                
                for a in v1.domain():
                    for b in v2.domain():
                        if a != b:
                            sat_tuples.append((a, b))
                            
                con.add_satisfying_tuples(sat_tuples)
                csp.add_constraint(con)

    return csp, var_array


def futoshiki_csp_model_2(futo_grid):

    n = len(futo_grid)
    var_array = []
    
    for i in range(n):
        rv = []
        for j in range(n):
            val = futo_grid[i][2*j]  
            
            if val == 0:
                domain = list(range(1, n+1))
            else:
                domain = [val]
                
            var = Variable(f"V({i},{j})", domain)
            rv.append(var)
        var_array.append(rv)

    # Create the CSP object
    csp = CSP("Futoshiki_Model_2")
    for i in range(n):
        for j in range(n):
            csp.add_var(var_array[i][j])
            
            
    
    # all row constraints

    for i in range(n):
        rv = var_array[i]
        all_diff = Constraint(f"RNEQ({i})", rv)
        domains = [v.domain() for v in rv]
        sat_tuples = []
        
        for cart_product in itertools.product(*domains):
            if len(set(cart_product)) == len(cart_product):
                sat_tuples.append(cart_product)
                
        all_diff.add_satisfying_tuples(sat_tuples)
        csp.add_constraint(all_diff)

    
        for j in range(n-1):
            sign = futo_grid[i][2*j + 1]
            
            if sign == '<':
                v_left  = var_array[i][j]
                v_right = var_array[i][j+1]
                con = Constraint(f"RL({i},{j})", [v_left, v_right])
                all_con = []
                
                for a in v_left.domain():
                    for b in v_right.domain():
                        if a < b:
                            all_con.append((a, b))
                            
                con.add_satisfying_tuples(all_con)
                csp.add_constraint(con)
                
            elif sign == '>':
                v_left  = var_array[i][j]
                v_right = var_array[i][j+1]
                con = Constraint(f"RG({i},{j})", [v_left, v_right])
                all_con = []
                
                for a in v_left.domain():
                    for b in v_right.domain():
                        if a > b:
                            all_con.append((a, b))
                    
                con.add_satisfying_tuples(all_con)
                csp.add_constraint(con)

    # all column constraints
    for j in range(n):
        col_vars = [var_array[i][j] for i in range(n)]
        all_diff = Constraint(f"CNEQ({j})", col_vars)
        domains = [v.domain() for v in col_vars]
        sat_tuples = []
        
        for cart_product in itertools.product(*domains):
            if len(set(cart_product)) == len(cart_product):
                sat_tuples.append(cart_product)
                
        all_diff.add_satisfying_tuples(sat_tuples)
        csp.add_constraint(all_diff)

    return csp, var_array
   