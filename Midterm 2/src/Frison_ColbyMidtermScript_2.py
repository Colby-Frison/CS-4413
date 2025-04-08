import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx
from datetime import datetime
import random

"""
File ran 03-28-2025 at 7:34 PM
"""

def question1_monte_carlo():
    """
    Question 1: Monte Carlo estimation of π/4
    """
    def generate_points(N):
        return np.random.uniform(0, 1, (N, 2))
    
    def estimate_pi_fourth(points):
        inside_circle = np.sum(np.sqrt(points[:, 0]**2 + points[:, 1]**2) <= 1)
        return inside_circle / len(points)
    
    # Part a: Generate and plot 1000 points
    N = 1000
    points = generate_points(N)
    
    plt.figure(figsize=(10, 10))
    plt.scatter(points[:, 0], points[:, 1], alpha=0.5)
    plt.title(f'Monte Carlo Points (N={N})')
    plt.xlabel('u')
    plt.ylabel('v')
    plt.axis('square')
    plt.grid(True)
    plt.savefig('../figures/Q1_monte_carlo_points.png')
    plt.close()
    
    # Part b: Estimate π/4 for different N values
    N_values = [10**3, 10**4, 10**5, 10**6]
    estimates = []
    
    for N in N_values:
        points = generate_points(N)
        pi_fourth_estimate = estimate_pi_fourth(points)
        estimates.append(pi_fourth_estimate)
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(N_values, estimates, 'o-')
    plt.axhline(y=np.pi/4, color='r', linestyle='--', label='True π/4')
    plt.title('Monte Carlo Estimation of π/4 vs Sample Size')
    plt.xlabel('Number of Points (N)')
    plt.ylabel('Estimated π/4')
    plt.grid(True)
    plt.legend()
    plt.savefig('../figures/Q1_pi_fourth_estimates.png')
    plt.close()
    
    return estimates

def question2_parenthesizing(n_max=20):
    """
    Question 2: Number of ways to parenthesize n atoms
    """
    # Initialize memoization array
    P = np.zeros(n_max + 1)
    P[1] = 1  # Base case: single atom
    
    # Calculate P(n) using the recurrence relation
    for n in range(2, n_max + 1):
        for k in range(1, n):
            P[n] += P[k] * P[n-k]
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, n_max + 1), P[2:], 'o-')
    plt.title('Number of Ways to Parenthesize vs Number of Atoms')
    plt.xlabel('Number of Atoms (n)')
    plt.ylabel('Number of Ways P(n)')
    plt.grid(True)
    plt.yscale('log')
    plt.savefig('../figures/Q2_parenthesizing.png')
    plt.close()
    
    return P[2:]

def p(n, j):
    """
    Calculate p(n,j): probability that j out of n folks will generate the number 1
    p(n,j) = C(n,j) * (1/n)^j * (1-1/n)^(n-j)
    """
    if j > n:
        return 0
    try:
        return math.comb(n, j) * pow(1/n, j) * pow(1-1/n, n-j)
    except OverflowError:
        return 0  # Return 0 for very large numbers that cause overflow

def theoretical_L(n, memo=None):
    """
    Calculate L(n) recursively using the formula from slides:
    L(n) = [1 + Σⱼ₌₂ⁿ⁻¹ L(j)p(n,j)] / [1 - p(n,0) - p(n,n)]
    Uses memoization to avoid recalculating values.
    """
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return 1
    
    # Calculate denominator: 1 - p(n,0) - p(n,n)
    denominator = 1 - p(n,0) - p(n,n)
    
    # Calculate sum term: Σⱼ₌₂ⁿ⁻¹ L(j)p(n,j)
    sum_term = sum(theoretical_L(j, memo) * p(n,j) for j in range(2, n))
    
    # Calculate L(n) using the formula from slides
    result = (1 + sum_term) / denominator
    
    memo[n] = result
    return result

def simulate_leader_election(n, num_trials=10000):
    """
    Simulate the Las Vegas leader election algorithm following the exact steps:
    Step 1: Everyone generates a random number in [1,n]
    Step 2: Count n1 people who got 1, if n1=1 we have winner
    Step 3: If n1>1, this subgroup generates numbers in [1,n1]
    Step 4: Continue until exactly one person gets 1
    """
    total_rounds = 0
    
    for _ in range(num_trials):
        rounds = 0
        people = n  # Current number of people
        
        while True:
            rounds += 1
            
            # Step 1: Everyone generates a number in [1,people]
            # Each person has 1/people chance of getting 1
            ones = sum(1 for _ in range(people) if random.randint(1, people) == 1)
            
            # Step 2: Check if we have a winner
            if ones == 1:  # Winner found
                break
            elif ones > 1:  # Multiple people got 1
                # Step 3: This subgroup continues
                people = ones
            # If ones == 0, repeat with same number of people
            
        total_rounds += rounds
    
    return total_rounds / num_trials

def plot_leader_election():
    """Plot the leader election results and verify the condition:
    lim(n→∞) L(n) < 2.442
    """
    n_values = list(range(2, 101))  # Extended to 100
    theoretical_values = []
    memo = {}  # Shared memoization dictionary
    
    # Calculate theoretical values with shared memoization
    for n in n_values:
        theoretical_values.append(theoretical_L(n, memo))
    
    plt.figure(figsize=(12, 7))
    plt.plot(n_values, theoretical_values, 'bo-', label='L(n)', markersize=3)
    plt.axhline(y=2.442, color='g', linestyle='--', label='limit = 2.442', alpha=0.5)
    plt.xlabel('n')
    plt.ylabel('L(n)')
    plt.title('Recursive function L(n) vs. n')
    plt.grid(True, alpha=0.3)  # Made grid lighter
    plt.ylim(2.0, 2.5)  # Set y-axis limits to match the image
    plt.legend()
    plt.savefig('../figures/leader_election.png')
    plt.close()
    
    # Print the values and verify conditions
    print("\nLeader Election Average Rounds:")
    print("n\tTheoretical\tL(n) < 2.442")
    max_L = float('-inf')
    for n, theo in zip(n_values, theoretical_values):
        max_L = max(max_L, theo)
        print(f"{n}\t{theo:.6f}\t{theo < 2.442}")
    
    # Print verification of condition
    print("\nVerifying condition:")
    print(f"lim(n→∞) L(n) < 2.442: {theoretical_values[-1]:.6f} < 2.442")

def question4_city_distances():
    """
    Question 4: City distances and graph algorithms
    """
    # Define city positions (x, y coordinates)
    cities = {
        'a': (0, 0),
        'b': (1, 1),
        'c': (0, 3),
        'd': (0, 2),
        'e': (1, 2),
        'f': (3, 1)
    }
    
    # Calculate distance matrix
    n = len(cities)
    distance_matrix = np.zeros((n, n))
    city_list = list(cities.keys())
    
    for i in range(n):
        for j in range(n):
            if i != j:
                x1, y1 = cities[city_list[i]]
                x2, y2 = cities[city_list[j]]
                distance_matrix[i][j] = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    # Create graph for MST
    G = nx.Graph()
    for i in range(n):
        for j in range(i+1, n):
            G.add_edge(city_list[i], city_list[j], weight=distance_matrix[i][j])
    
    # Compute MST using Prim's algorithm
    mst_prim = nx.minimum_spanning_tree(G, algorithm='prim')
    
    # Compute MST using Kruskal's algorithm
    mst_kruskal = nx.minimum_spanning_tree(G, algorithm='kruskal')
    
    # Compute all-pairs shortest paths
    shortest_paths = nx.floyd_warshall(G)
    
    # Visualize the complete graph
    plt.figure(figsize=(15, 6))
    
    # Plot original graph
    plt.subplot(121)
    pos = cities  # Use the city coordinates for positioning
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=16, font_weight='bold')
    
    # Format edge labels to 2 decimal places and adjust label positions
    edge_labels = {}
    for (u, v, d) in G.edges(data=True):
        # Format weight to 2 decimal places
        edge_labels[(u, v)] = f'{d["weight"]:.2f}'
    
    # Draw edge labels with adjusted position and background
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=8,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
        label_pos=0.3  # Adjust label position along the edge
    )
    plt.title('Complete Graph')
    
    # Plot MST
    plt.subplot(122)
    nx.draw(mst_prim, pos, with_labels=True, node_color='lightgreen',
            node_size=500, font_size=16, font_weight='bold')
    
    # Format MST edge labels
    edge_labels = {}
    for (u, v, d) in mst_prim.edges(data=True):
        edge_labels[(u, v)] = f'{d["weight"]:.2f}'
    
    nx.draw_networkx_edge_labels(
        mst_prim, pos,
        edge_labels=edge_labels,
        font_size=10,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
        label_pos=0.5
    )
    plt.title('Minimum Spanning Tree')
    
    # Add more spacing between subplots
    plt.tight_layout(pad=3.0)
    
    plt.savefig('../figures/Q4_city_graph_and_mst.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    return distance_matrix, mst_prim, mst_kruskal, shortest_paths

def question5_stirling():
    """
    Question 5: Stirling's approximation analysis
    """
    def stirling(n):
        return np.sqrt(2 * np.pi * n) * (n/np.e)**n
    
    def factorial(n):
        return math.factorial(n)
    
    n_values = range(2, 21)
    factorials = [factorial(n) for n in n_values]
    stirling_approx = [stirling(n) for n in n_values]
    
    # Absolute error
    errors = [f - s for f, s in zip(factorials, stirling_approx)]
    
    # Relative error
    rel_errors = [e/f for e, f in zip(errors, factorials)]
    
    # Plot absolute error
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, errors, 'o-')
    plt.title("Absolute Error in Stirling's Approximation")
    plt.xlabel('n')
    plt.ylabel('Error e(n)')
    plt.grid(True)
    plt.savefig('../figures/Q5_stirling_error.png')
    plt.close()
    
    # Plot relative error
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, rel_errors, 'o-')
    plt.title("Relative Error in Stirling's Approximation")
    plt.xlabel('n')
    plt.ylabel('Relative Error re(n)')
    plt.grid(True)
    plt.savefig('../figures/Q5_stirling_relative_error.png')
    plt.close()
    
    return errors, rel_errors

def main():
    # Record start time
    start_time = datetime.now()
    print(f"Starting calculations at {start_time}")
    
    # Run all questions
    print("\nQuestion 1: Monte Carlo π/4 Estimation")
    pi_fourth_estimates = question1_monte_carlo()
    print("π/4 estimates for N=1000, 10000, 100000, 1000000:")
    for N, est in zip([10**3, 10**4, 10**5, 10**6], pi_fourth_estimates):
        print(f"N={N}: π/4 ≈ {est:.6f}")
    
    print("\nQuestion 2: Parenthesizing")
    parenthesizing_results = question2_parenthesizing()
    print("Number of ways to parenthesize (first few values):")
    for n, p in enumerate(parenthesizing_results[:5], start=2):
        print(f"P({n}) = {int(p)}")
    
    print("\nQuestion 3: Leader Election")
    plot_leader_election()
    
    print("\nQuestion 4: City Distances")
    dist_matrix, mst_prim, mst_kruskal, shortest_paths = question4_city_distances()
    print("Distance matrix:")
    print(dist_matrix)
    print("\nMST cost (Prim):", sum(d['weight'] for (u, v, d) in mst_prim.edges(data=True)))
    print("MST cost (Kruskal):", sum(d['weight'] for (u, v, d) in mst_kruskal.edges(data=True)))
    
    print("\nQuestion 5: Stirling's Approximation")
    errors, rel_errors = question5_stirling()
    print("First few absolute errors:")
    for n, e in enumerate(errors[:5], start=2):
        print(f"e({n}) = {e:.6f}")
    print("\nFirst few relative errors:")
    for n, re in enumerate(rel_errors[:5], start=2):
        print(f"re({n}) = {re:.6f}")
    
    # Record end time
    end_time = datetime.now()
    print(f"\nCalculations completed at {end_time}")

if __name__ == "__main__":
    main() 