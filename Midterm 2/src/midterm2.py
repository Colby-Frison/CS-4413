import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx
from datetime import datetime

def question1_monte_carlo():
    """
    Question 1: Monte Carlo estimation of π/4
    """
    def generate_points(N):
        return np.random.uniform(0, 1, (N, 2))
    
    def estimate_pi(points):
        inside_circle = np.sum(np.sqrt(points[:, 0]**2 + points[:, 1]**2) <= 1)
        return 4 * inside_circle / len(points)
    
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
    plt.savefig('monte_carlo_points.png')
    plt.close()
    
    # Part b: Estimate π/4 for different N values
    N_values = [10**3, 10**4, 10**5, 10**6]
    estimates = []
    
    for N in N_values:
        points = generate_points(N)
        pi_estimate = estimate_pi(points)
        estimates.append(pi_estimate)
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(N_values, estimates, 'o-')
    plt.axhline(y=np.pi, color='r', linestyle='--', label='True π')
    plt.title('Monte Carlo Estimation of π vs Sample Size')
    plt.xlabel('Number of Points (N)')
    plt.ylabel('Estimated π')
    plt.grid(True)
    plt.legend()
    plt.savefig('pi_estimates.png')
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
    plt.savefig('parenthesizing.png')
    plt.close()
    
    return P[2:]

def question3_leader_election(n_max=20):
    """
    Question 3: Average rounds needed for leader election
    """
    # Initialize array for average rounds
    L = np.zeros(n_max + 1)
    L[1] = 0  # Base case: with 1 person, no rounds needed
    
    # Calculate L(n) using the recurrence relation
    for n in range(2, n_max + 1):
        # L(n) = 1 + (n-1)/n * L(n-1)
        L[n] = 1 + ((n-1)/n) * L[n-1]
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, n_max + 1), L[2:], 'o-')
    plt.title('Average Rounds Needed for Leader Election')
    plt.xlabel('Number of People (n)')
    plt.ylabel('Average Number of Rounds L(n)')
    plt.grid(True)
    plt.savefig('leader_election.png')
    plt.close()
    
    return L[2:]

def question4_city_distances():
    """
    Question 4: City distances and graph algorithms
    """
    # Define city positions (x, y coordinates)
    cities = {
        'a': (0, 0),
        'b': (1, 0),
        'c': (2, 0),
        'd': (0, 1),
        'e': (1, 1),
        'f': (2, 1)
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
    plt.savefig('stirling_error.png')
    plt.close()
    
    # Plot relative error
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, rel_errors, 'o-')
    plt.title("Relative Error in Stirling's Approximation")
    plt.xlabel('n')
    plt.ylabel('Relative Error re(n)')
    plt.grid(True)
    plt.savefig('stirling_relative_error.png')
    plt.close()
    
    return errors, rel_errors

def main():
    # Record start time
    start_time = datetime.now()
    print(f"Starting calculations at {start_time}")
    
    # Run all questions
    print("\nQuestion 1: Monte Carlo π Estimation")
    pi_estimates = question1_monte_carlo()
    print("π estimates for N=1000, 10000, 100000, 1000000:")
    for N, est in zip([10**3, 10**4, 10**5, 10**6], pi_estimates):
        print(f"N={N}: π ≈ {est:.6f}")
    
    print("\nQuestion 2: Parenthesizing")
    parenthesizing_results = question2_parenthesizing()
    print("Number of ways to parenthesize (first few values):")
    for n, p in enumerate(parenthesizing_results[:5], start=2):
        print(f"P({n}) = {int(p)}")
    
    print("\nQuestion 3: Leader Election")
    leader_results = question3_leader_election()
    print("Average rounds needed (first few values):")
    for n, l in enumerate(leader_results[:5], start=2):
        print(f"L({n}) = {l:.4f}")
    
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
    print(f"Total runtime: {end_time - start_time}")

if __name__ == "__main__":
    main() 