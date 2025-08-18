import gurobipy as gp
from gurobipy import GRB
import itertools


def find_BFS(M, G, E, utils, p):
    """
    Finds a BFS of proportional polytope maximizing the utilitarian welfare.

    This function determines a fractional allocation of goods to groups of agents
    with the goal of maximizing the total utility (utilitarian social welfare).
    The allocation is subject to two main constraints: every good must be fully
    allocated, and every agent must receive at least a predefined minimum
    utility.

    Args:
        M (list): A list of the goods to be allocated, e.g., [0, 1, 2, ...].
        G (list[list[int]]): Configuration of groups where G[g] contains the agents in group g.
        E (list[tuple]): A list of valid (good, group) tuples representing the
            edges in the allocation graph. An allocation can only happen along
            these edges.
        utils (list[list[float]]): A 2D list where utils[agent][good] is the
            utility of a good for an agent.
        p (dict): A dictionary where p[(g, i)] is the minimum utility that
            (g, i) must receive.

    Returns:
        str or dict:
            - "infeasible" if no allocation satisfying the minimum utility
              constraints can be found.
            - A dictionary representing the fractional allocation, where keys
              are edges (good, group) and values are the fraction of the
              good allocated to that group (e.g., {(good, group): 0.5, ...}).
    """
    n = len(G)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model(env=env)

    # x[a, g]] is a continuous variable representing the fraction of good 'a'
    # that is allocated to group 'g'.
    x = model.addVars([e for e in E], lb=0, ub=1, vtype=GRB.CONTINUOUS, name="x")
    
    # u[g, i] represents the total utility for the i-th agent in group g.
    # This is used for setting the objective function.
    u = model.addVars([(g, i) for g in range(n) for i in range(len(G[g]))],
                      vtype=GRB.CONTINUOUS,
                      name="u")

    # Constraint: Each good must be fully (fractionally) allocated.
    for a in M:
        model.addConstr(gp.quicksum(x[a, g] for g in range(n) if (a, g) in E) == 1)

    # Constraint: The utility received by each agent must meet their minimum requirement.
    for g in range(n):
        for i in range(len(G[g])):
            agent = G[g][i]
            model.addConstr(gp.quicksum(utils[agent][a] * x[a, g] for a in M if (a, g) in E) >= p[(g, i)])

    # Link the utility variables 'u' to the calculated utilities for the objective.
    # The small epsilon prevents potential numerical issues, though it's less
    # critical here since the objective is linear, not logarithmic.
    for g in range(n):
        for i in range(len(G[g])):
            agent = G[g][i]
            model.addConstr(u[g, i] == gp.quicksum(utils[agent][a] * x[a, g] for a in M if (a, g) in E))

    # Objective: Maximize the sum of all agents' utilities (utilitarian welfare).
    model.setObjective(gp.quicksum(u[g, i] for g in range(n) for i in range(len(G[g]))), GRB.MAXIMIZE)

    model.setParam("Method", 0)  # Use simplex as the solver to find an optimal BFS
    model.optimize()

    # Check the result of the optimization.
    if model.status == GRB.INFEASIBLE:
        return "infeasible"
    
    # Store the resulting fractional allocation in a dictionary.
    res = {}
    for e in E:
        res[e] = x[e].X
    return res


def iter_alg(G, m, utils, remove_the_best=0):
    """
    Implements an iterative rounding algorithm to find a almost proportional
    allocation of goods to groups of agents.

    Please read section 4.1 of the paper for a detailed discription of the algorithm.

    Args:
        G (list[list[int]]): Cofiguration of groups where G[g] contains the agents in group g.
            This list is modified by the algorithm as agents are removed.
        m (int): The total number of goods.
        utils (list[list[float]]): A 2D list where utils[agent][good] is the
            utility of a good for an agent.
        remove_the_best (int): A flag that controls the agent removal strategy.
            0: Removes the last agent of any group with low incident weight 
                in the fractional solution.
            1: Finds the group with the minimum incident weight in the fractional solution
            and removes the agent who has larger utility for the goods already allocated to that group.

    Returns:
        list[list[int]]: A list representing the final integral allocation,
            where B[g] contains the list of goods assigned to group g.
    """
    n = len(G)
    M = list(range(m))  # The set of remaining goods to be allocated.
    E = list(itertools.product(M, range(n))) # The set of all possible (good, group) allocation edges.
    
    # Initialize p: the minimum utility requirement for each agent.
    # Initially, this is set to their proportional share (1/n of total utility).
    p = {}
    for g in range(n):
        for i in range(len(G[g])):
            agent = G[g][i]
            p[(g, i)] = sum(utils[agent]) / n
            
    # B will store the final integral allocation of goods to groups.
    B = [[] for _ in range(n)]

    # The main loop continues as long as there are goods left to allocate.
    while len(M) > 0:
        made_progress = False
        # Find a fractional allocation based on the current state (M, G, E, p).
        x = find_BFS(M, G, E, utils, p)
        

        # Round the fractional solution. We first round the integral edges.
        E2 = E.copy()
        for e in E:
            a = e[0] # good
            g = e[1] # group
            if x[e] == 0:
                # remove the edge.
                E2.remove(e)
                made_progress = True
            elif x[e] == 1:
                # Assign good a to group g
                E2.remove(e)
                B[g].append(a)
                M.remove(a)
                made_progress = True
        E = E2.copy()
        
        # --- Agent Removal Strategy ---
        if not remove_the_best:
            # Strategy 0: Remove the last agent of ANY group with low incident weight in x.
            for g in range(n):
                # This strategy uses E and M for making its decision
                if len(G[g]) > 0 and sum(x[(a, g)] for a in M if (a, g) in E) <= len(G[g]):
                    G[g].pop() # Remove the last agent in the group list.
                    made_progress = True
        else:
            # Strategy 1: Find the group with the minimum incident weight in x
            # and removes the agent who has the most utility for the goods already allocated to that group.
            min_g = -1 # The index of the group with the minimum incident weight in x
            # Find the first group with low incident weight
            for g in range(n):
                if len(G[g]) > 0 and sum(x[(a, g)] for a in M if (a, g) in E) <= len(G[g]):
                    min_g = g
                    break
            # Find if there is a group with even lower incident weight
            if min_g != -1:
                for g in range(n):
                    if len(G[g]) > 0 and sum(x[(a, g)] for a in M if (a, g) in E) <= len(G[g]) and sum(x[(a, g)] for a in M if (a, g) in E) < sum(x[(a, min_g)] for a in M if (a, min_g) in E):
                        min_g = g
                
                # Decide which agent to remove from the chosen group 'min_g'.
                ind = 0 # Default to removing the first agent.
                if len(G[min_g]) == 2:
                    # If there are two agents, remove the one who has larger utility
                    # for the goods already allocated to this group.
                    agent0_util = sum(utils[G[min_g][0]][a] for a in B[min_g])
                    agent1_util = sum(utils[G[min_g][1]][a] for a in B[min_g])
                    if agent0_util < agent1_util:
                        ind = 1
                G[min_g].pop(ind)
                made_progress = True
                
        # Assert that the algorithm is not stuck in an infinite loop.
        assert made_progress
        
        # Update the minimum utility requirements for the next iteration.
        for g in range(n):
            for i in range(len(G[g])):
                agent = G[g][i]
                # The new requirement is the original proportional share minus the
                # utility of goods they have already received.
                p[(g, i)] = sum(utils[agent]) / n - sum(utils[agent][a] for a in B[g])
                
    return B
