from iterative_rounding import *
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

import random
import copy
from pathlib import Path


def buildModel(N, M, utils, G, arg):
    """
    Builds and solves a Gurobi integer programming model to determine if a fair
    allocation of goods exists for the given fair divison instance

    The model allocates M goods to N groups, where each group consists of two
    agents. The goal is to find an allocation that satisfies a specified
    fairness criterion (EF, EF1, or EFX) for every agent in every group.

    Args:
        N (int): The number of groups.
        M (int): The number of goods to be allocated.
        utils (list[list[float]]): A 2D list where utils[agent][good] is the utility
            the good for the agent.
        G (list[list[int]]): A list containing the cofiguration of groups.
            G[g] contains the indicesof the two agents forming group g.
        arg (str): The fairness criterion to enforce. Must be one of:
            "EF"  - Envy-Free
            "EF1" - Envy-Free up to one good
            "EFX" - Envy-Free up to any good

    Returns:
        int: 1 if a fair allocation is found (model is feasible),
             0 otherwise (model is infeasible).
    """
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model(env=env)

    # x[i, k] indicates whether good a is allocated to group g
    x = model.addVars(
        [
            (g, a)
            for g in range(N)
            for a in range(M)
        ],
        vtype=GRB.BINARY,
        name="x",
    )

    # 'removed' variables are only needed for the EF1 criterion.
    # removed[g, i, g2, a] = 1 means good a must be removed from g2's
    # bundle to satisfy envy-freeness for agent (g, i).
    if arg == "EF1":
        removed = model.addVars(
            [
                (g, i, g2, a)
                for g in range(N)
                for i in range(2)
                for g2 in range(N)
                for a in range(M + 1)
                # k=M is a dummy good representing the case where j's bundle is empty
                # and no good needs to be removed.
            ],
            vtype=GRB.BINARY,
            name="removed_good",
        )

    # Constraint: Each good must be allocated to exactly one group.
    for a in range(M):
        model.addConstr(gp.quicksum(x[i, a] for i in range(N)) == 1)

    # Fairness Constraints:
    # Iterate through each group i and each potential envied group j.
    for g in range(N):
        for g2 in range(N):
            if g != g2:
                # Apply the constraint for both agents (g, i).
                for i in range(2):  # agent G[i][i] must not envy group j
                    agent_id = G[g][i]
                    
                    # EF: The agent's utility for their own bundle must be >=
                    # their utility for the other group's bundle.
                    if arg == "EF":
                        model.addConstr(
                            gp.quicksum(utils[agent_id][a] * x[g, a] for a in range(M)) >=
                            gp.quicksum(utils[agent_id][a] * x[g2, a] for a in range(M))
                        )
                    
                    # EF1: There must exist at least one good in g2's bundle whose
                    # removal eliminates envy.
                    elif arg == "EF1":
                        # A good a can only be the 'removed' good if it's actually in g2's bundle.
                        for a in range(M):
                            model.addConstr(removed[g, i, g2, a] <= x[g2, a])
                        
                        # If we designate a as the good to remove, the EF condition must hold
                        # for the remaining bundle.
                        for a in range(M):
                            model.addConstr((removed[g, i, g2, a] == 1) >> (
                                gp.quicksum(utils[agent_id][b] * x[g, b] for b in range(M)) >=
                                gp.quicksum(utils[agent_id][b] * x[g2, b] for b in range(M)) - utils[agent_id][a]
                            ))
                        
                        # If g2's bundle is empty, no good needs to be removed.
                        model.addConstr((removed[g, i, g2, M] == 1) >> (
                            gp.quicksum(utils[agent_id][a] * x[g, a] for a in range(M)) >=
                            gp.quicksum(utils[agent_id][a] * x[g2, a] for a in range(M))
                        ))

                        # At least one 'removed' option (a real good or the dummy good) must be chosen.
                        model.addConstr(gp.quicksum(removed[g, i, g2, a] for a in range(M + 1)) >= 1)

                    # EFX: For any good a in g2's bundle, its removal must eliminate envy.
                    elif arg == "EFX":
                        for a in range(M):
                            # This implication only triggers if good a is in g2's bundle
                            # and has positive utility for the agent.
                            if utils[agent_id][a] > 0:
                                model.addConstr((x[g2, a] == 1) >> (
                                    gp.quicksum(utils[agent_id][b] * x[g, b] for b in range(M)) >=
                                    gp.quicksum(utils[agent_id][b] * x[g2, b] for b in range(M)) - utils[agent_id][a]
                                ))
                    else:
                        raise ValueError("Invalid fairness criterion specified.")
                    
    model.optimize()
    # Check the result of the optimization.
    if model.status == GRB.INFEASIBLE:
        return 0  # No fair allocation found
    return 1  # A fair allocation exists



def create_pairs(N, partner, single):
    """
    Recursively generates all possible pairings for a set
    of N agents.

    This function finds all unpaired agents (initially marked as -1) and 
    explores all valid ways to pair them up until every agent is paired. 
    When the number of agents is odd, one agent will be paired with themself.

    Args:
        N (int): The total number of agents, indexed from 0 to N-1.
        partner (list[int]): A list representing the current pairing state. 
            partner[i] = j indicates that agent i is paired with agent j. 
            An unpaired agent is marked with -1. This list is modified during 
            recursion and should be restored to its original state after the call.
        single (bool): A flag indicating whether an unpaired agent can be
            paired with themself. That is, whether the number of remaining agents 
            is Odd.

    Returns:
        list[list[list[int]]]: A list of all possible pairings.
            Each pairing is represented as a list of pairs,
            e.g., [[[0, 1], [2, 3]], [[0, 2], [1, 3]]].
    """
    # Base Case: If no agents are marked as -1, a complete pairing is found.
    if -1 not in partner:
        res = []
        # Build the list of pairs.
        for i in range(N):
            if i <= partner[i]:
                res.append([i, partner[i]])
        # Return the single complete matching found in this recursive branch.
        return [res]

    res = []
    # Find the first unpaired agent to start the next pairing.
    for i in range(N):
        if partner[i] == -1:
            # Option 1: Pair the agent with themself if the 'single' flag is set.
            if single:
                partner[i] = i  # Pair i with itself.
                # Recursively find all pairings for the remaining agents.
                res.extend(create_pairs(N, partner, False))
                partner[i] = -1  # Backtrack: un-pair i.

            # Option 2: Pair agent i with another unpaired agent j.
            for j in range(i + 1, N):
                if partner[j] == -1:
                    # Pair i and j together.
                    partner[i] = j
                    partner[j] = i
                    # Recursively find all pairings for the remaining agents.
                    res.extend(create_pairs(N, partner, single))
                    # Backtrack: un-pair i and j.
                    partner[i] = partner[j] = -1
            
            # Once we have processed the first unpaired agent 'i', we can return
            # the results, as all subsequent recursive calls will handle the rest.
            return res


def prop(alloc, G, m, utils, k):
    """
    Checks if an allocation satisfies a "proportional up to k items" fairness criterion.

    Args:
        alloc (list[list[int]]): A list where alloc[g] contains the goods
            allocated to group g.
        G (list[list[int]]): Configuration of groups where G[g] contains the agents in group g.
        m (int): The total number of goods.
        utils (list[list[float]]): A 2D list where utils[agent][good] is the
            utility of a good for an agent.
        k (list[int]): A list of integers where we want each agent (g, i) to be propk.

    Returns:
        int: 1 if the allocation satisfies the property for all agents,
             0 otherwise.
    """
    M = list(range(m))
    n = len(G)
    # Check for all groups and all agents in each group
    for g in range(n):
        # Goods not allocated to g
        other_goods = [b for b in M if b not in alloc[g]]
        for i in range(len(G[g])):
            agent = G[g][i]
            # Get the agent utilities for the goods they were not allocated.
            sorted_utils = sorted([utils[agent][a] for a in other_goods])
            
            # Check if the agent's current utility plus the utility of the top k
            # other goods meets their proportional share.
            current_utility = sum(utils[agent][a] for a in alloc[g])
            top_k_utility = sum(sorted_utils[len(sorted_utils) - k[i]:])
            proportional_share = sum(utils[agent]) / n
            if current_utility + top_k_utility < proportional_share:
                return 0 
    return 1

def ef(alloc, G, utils, arg):
    """
    Checks if an allocation satisfies a specified envy-freeness criterion.

    This function can check for standard Envy-Freeness (EF), Envy-Freeness up
    to one good (EF1), or Envy-Freeness up to any good (EFX).

    Args:
        alloc (list[list[int]]): A list where alloc[g] contains the goods
            allocated to group g.
        G (list[list[int]]): onfiguration of groups where G[g] contains the agents in group g.
        utils (list[list[float]]): A 2D list where utils[agent][good] is the
            utility of a good for an agent.
        arg (int, str, or None): The fairness criterion to check.
            None: Checks for standard Envy-Freeness (EF).
            1:    Checks for Envy-Freeness up to one good (EF1).
            'X':  Checks for Envy-Freeness up to any good (EFX).

    Returns:
        int: 1 if the allocation is fair according to the specified criterion,
             0 otherwise.
    """
    n = len(G)
    # Check for all groups and all agents in each group
    for g in range(n):
        for i in range(len(G[g])):
            agent = G[g][i]
            agent_bundle_utility = sum(utils[agent][a] for a in alloc[g])
            
            for g2 in range(n):
                # An agent doesn't envy their own group or a group with empty bundle.
                if g == g2 or len(alloc[g2]) == 0:
                    continue
                
                other_bundle_utility = sum(utils[agent][a] for a in alloc[g2])

                # EF1 Check
                if arg == 1:
                    # Find the good in the other bundle most valued by the current agent.
                    most_valued_good_utility = max(utils[agent][a] for a in alloc[g2])
                    # If envy persists even after removing this best good, it's not EF1.
                    if agent_bundle_utility < other_bundle_utility - most_valued_good_utility:
                        return 0
                
                # EFX Check
                elif arg == 'X':
                    # Check for every good in the other bundle.
                    for b in alloc[g2]:
                        # If removing any single good b (with positive utility)
                        # does not eliminate envy, it's not EFX.
                        if utils[agent][b] > 0 and agent_bundle_utility < other_bundle_utility - utils[agent][b]:
                            return 0
                
                # EF Check
                else:
                    if agent_bundle_utility < other_bundle_utility:
                        return 0
    return 1





def main():
    # Pre-compute all possible ways to form pairs for groups of agents from size 0 to 15.
    all_group_combs = []
    for n in range(16):
        # The `single` flag is set to True only for odd n, to pair one of the agents with themself
        all_group_combs.append(list(create_pairs(n, [-1 for _ in range(n)], bool(n % 2))))

    directory_path = Path(r"./spliddit")
    random.seed(0)
    # Initialize a list to store the results of all experiments.
    data = []
    # Iterate Through Data Files ---
    # Process each problem instance file in the specified directory.
    for filename in directory_path.glob("*.instance"):
        with open(filename, 'r') as f:
            lines = f.readlines()
            # Parse the number of agents (N) and goods (M) from the file.
            N, M = [int(x) for x in lines[0].split()]
            # Parse the utility matrix from the file.
            utils = []
            for i in range(2, N + 2):
                utils.append([int(x) for x in lines[i].split()])

            # Only run experiments for instances with at least 4 agents.
            if N > 3:
                # If the number of different pairings in the instance is more than 1000, 
                # kepp only 1000 random pairirngs.
                pairings = random.sample(all_group_combs[N], k=min(1000, len(all_group_combs[N])))

                # --- Experiment 1: Check for Existence of Fair Allocations ---
                count_EF1 = count_EF = count_EFX = 0
                for i, G in enumerate(pairings):
                    # Use the Gurobi model to check if a solution exists for each fairness notion.
                    # (N + 1) // 2 is the number of groups
                    count_EF1 += buildModel((N + 1) // 2, M, utils, G, "EF1")
                    count_EF += buildModel((N + 1) // 2, M, utils, G, "EF")
                    count_EFX += buildModel((N + 1) // 2, M, utils, G, "EFX")
                # Record the results for this experiment. As we realized ef1 exists for all agents, we did not implement
                # a function to check prop1.
                data.append({"Instance": filename, "N": N, "M": M, "Total": len(pairings), "Algorithm": "existence",
                            "EF1": count_EF1, "EFX": count_EFX, "EF": count_EF, "PROP1": len(pairings)})

                # --- Experiment 2: Run the Algorithm "remove_all_agents" ---
                count_EF1 = count_EF = count_EFX = count_PROP1 = 0
                for i, G in enumerate(pairings):
                    # Run the iterative algorithm to get a concrete allocation.
                    alloc = iter_alg(copy.deepcopy(G), M, utils, remove_the_best=0)
                    # Sanity check: ensure the algorithm's output always meets its theoretical guarantee of prop1/prop2
                    assert prop(alloc, G, M, utils, [1, 2]) == 1, "Not Prop " + str(filename) + ", " + str(G) + ", " + str(alloc)
                    # Count whether the generated allocation satisfies various fairness criteria.
                    count_PROP1 += prop(alloc, G, M, utils, [1, 1])
                    count_EF1 += ef(alloc, G, utils, 1)
                    count_EFX += ef(alloc, G, utils, 'X')
                    count_EF += ef(alloc, G, utils, None)
                # Record the results for this experiment.
                data.append(
                    {"Instance": filename, "N": N, "M": M, "Total": len(pairings), "Algorithm": "remove all the agents",
                     "EF1": count_EF1, "EFX": count_EFX, "EF": count_EF, "PROP1": count_PROP1})
                
                
                # --- Experiment 3: Run the Heuristic Implementation "remove_the_best_agent" ---
                count_EF1 = count_EF = count_EFX = count_PROP1 = 0
                for i, G in enumerate(pairings):
                    # Run the iterative algorithm with a different setting.
                    alloc = iter_alg(copy.deepcopy(G), M, utils, remove_the_best=1)
                    # Sanity check: ensure the algorithm's output always meets its theoretical guarantee
                    # As in the huristic implimentation we may remove the second agent first, 
                    # we check for both prop1/prop2 and prop2/prop1
                    assert prop(alloc, G, M, utils, [1, 2]) == 1 or prop(alloc, G, M, utils, [2, 1]) == 1, "Not Prop " + str(filename) + ", " + str(G) + ", " + str(alloc)
                    # Count whether the generated allocation satisfies various fairness criteria.
                    count_PROP1 += prop(alloc, G, M, utils, [1, 1])
                    count_EF1 += ef(alloc, G, utils, 1)
                    count_EFX += ef(alloc, G, utils, 'X')
                    count_EF += ef(alloc, G, utils, None)
                # Record the results for this experiment.
                data.append(
                    {"Instance": filename, "N": N, "M": M, "Total": len(pairings), "Algorithm": "remove the best agent",
                     "EF1": count_EF1, "EFX": count_EFX, "EF": count_EF, "PROP1": count_PROP1})

                # --- Save and Display Results ---
                df = pd.DataFrame(data)
                df.to_csv('result.csv', index=False)
                print(df)


# Standard Python entry point.
if __name__ == "__main__":
    main()