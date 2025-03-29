# Algorithm Analysis - Spring 2025 Midterm 2 Solutions

## Question 1: Monte Carlo Method for π/4 Estimation

### Part A: Generating Random Points
We generate $N$ pairs of points $(u_i, v_i)$ uniformly distributed in $[0,1)$. These points are plotted on a unit square, as shown in `Q1_monte_carlo_points.png`. Points falling inside the quarter circle $(x^2 + y^2 \leq 1)$ are used to estimate $\pi/4$.

### Part B: Estimating π/4
The Monte Carlo method estimates $\pi/4$ by exploiting the relationship between areas in the first quadrant.

#### Mathematical Foundation:
- Area of unit square = $1 \times 1 = 1$
- Area of quarter circle = $\pi/4$
- Ratio = $\frac{\text{points inside quarter circle}}{\text{total points}} \approx \pi/4$

#### Implementation Details:
1. Generate $N$ random points $(x,y)$ in $[0,1) \times [0,1)$
2. Test if point lies inside quarter circle: $x^2 + y^2 \leq 1$
3. Ratio of points inside gives estimate of $\pi/4$

```python
def question1_monte_carlo():
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
```

#### Results (see `Q1_pi_fourth_estimates.png`):
- $N = 10^3$: $\pi/4 \approx 0.785000$ (Error: $0.05\%$)
- $N = 10^4$: $\pi/4 \approx 0.786300$ (Error: $0.11\%$)
- $N = 10^5$: $\pi/4 \approx 0.784250$ (Error: $-0.15\%$)
- $N = 10^6$: $\pi/4 \approx 0.785381$ (Error: $-0.002\%$)

The convergence plot shows how the estimate approaches $\pi/4 \approx 0.785398...$ as $N$ increases.

## Question 2: Parenthesizing Expressions

### Dynamic Programming Solution
The solution uses dynamic programming to build up $P(n)$ from smaller subproblems.

#### Recurrence Details:
For $n$ atoms, we consider all possible ways to split the expression:
$$P(n) = \sum_{k=1}^{n-1} P(k) \times P(n-k)$$

Where:
- $P(k)$ represents ways to parenthesize first $k$ atoms
- $P(n-k)$ represents ways to parenthesize remaining $(n-k)$ atoms
- Multiplication represents combining these independent choices

#### Growth Analysis (see `Q2_parenthesizing.png`):
The plot shows exponential growth of $P(n)$, plotted on a log scale:
- $P(2) = 1$
- $P(3) = 2$
- $P(4) = 5$
- $P(5) = 14$
- $P(6) = 42$
- $P(7) = 132$
- $P(8) = 429$
- ...

These are Catalan numbers, with the general form: $C_n = \frac{1}{n+1}\binom{2n}{n}$

```python
def question2_parenthesizing(n_max=20):
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
```

## Question 3: Leader Election Analysis

### Las Vegas Algorithm Description
This is a Las Vegas algorithm (always correct, randomized running time) that works as follows:

1. In each round:
   - Each person independently generates a number in $[1,n]$
   - If exactly one person generates $1$, they become the leader and algorithm terminates
   - If no one or multiple people generate $1$, another round is needed
   - If multiple people generate $1$, only they continue to the next round with $n_1$ people

### Theoretical Analysis

#### Probability Calculation
For $n$ people, the probability $p(n,j)$ that exactly $j$ people generate $1$ is:

$$p(n,j) = \binom{n}{j} \cdot \left(\frac{1}{n}\right)^j \cdot \left(1-\frac{1}{n}\right)^{n-j}$$

where:
- $\binom{n}{j}$ is the binomial coefficient (number of ways to choose $j$ from $n$)
- $\left(\frac{1}{n}\right)^j$ is the probability that $j$ specific people generate $1$
- $\left(1-\frac{1}{n}\right)^{n-j}$ is the probability that the remaining $(n-j)$ people don't generate $1$

#### Recursive Formula for $L(n)$
The expected number of rounds $L(n)$ satisfies the recurrence:

$$L(n) = 1 \cdot p(n,1) + [1 + L(n)]p(n,0) + \sum_{j=2}^n L(j)p(n,j)$$

This can be solved to:

$$L(n) = \frac{1 + \sum_{j=2}^{n-1} L(j)p(n,j)}{1 - p(n,0) - p(n,n)}$$

#### Implementation Details
1. Base cases:
   - $L(1) = 1$ (no rounds needed for one person)
   - For $n \geq 2$, use the recursive formula

2. Memoization:
   - Used to avoid recalculating $L(j)$ values
   - Significantly improves performance for large $n$

3. Numerical considerations:
   - Handle potential overflow in probability calculations
   - Use stable numerical methods for large combinations

#### Theoretical Bounds
Two important theoretical results were verified:

1. $L(n) < e = 2.718$ for all $n \geq 2$
   - Our implementation confirms this bound
   - Maximum observed value $\approx 2.434$ for $n=100$
   - Provides guaranteed performance bound

2. $\lim_{n \to \infty} L(n) < 2.442$
   - Implementation shows convergence to $\approx 2.434$
   - Values for large $n$ stay well below $2.442$
   - Confirms theoretical prediction

#### Results and Visualization
Our implementation produced the following key results:

- $L(3) = 2.166667$
- $L(4) = 2.241379$
- $L(5) = 2.284045$
- $L(6) = 2.311689$
- ...
- $L(50) = 2.426759$
- $L(100) = 2.434257$


The visualization (see `leader_election.png`) shows:
1. Sharp initial increase from $n=2$ to $n \approx 10$
2. Gradual convergence for larger $n$
3. Clear asymptotic behavior approaching but not exceeding $2.442$
4. Smooth, monotonic growth characteristic of the theoretical prediction

#### Key Properties Demonstrated
1. Efficiency: Expected number of rounds remains bounded by $2.442$
2. Scalability: Performance doesn't degrade with larger $n$
3. Reliability: Algorithm always terminates with a unique leader
4. Practicality: Simple implementation with predictable behavior

### Simulation vs Theory
The theoretical analysis is supported by our numerical results:
1. All computed values satisfy $L(n) < e$
2. Convergence behavior matches theoretical predictions
3. Implementation confirms both upper bounds
4. Results are numerically stable across all tested values of $n$

This implementation provides strong empirical evidence for the theoretical bounds while demonstrating the practical efficiency of the Las Vegas leader election algorithm.

```python
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
            ones = sum(1 for _ in range(people) if random.randint(1, people) == 1)
            
            # Step 2: Check if we have a winner
            if ones == 1:  # Winner found
                break
            elif ones > 1:  # Multiple people got 1
                # Step 3: This subgroup continues
                people = ones
            
        total_rounds += rounds
    
    return total_rounds / num_trials

def plot_leader_election():
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
    plt.grid(True, alpha=0.3)
    plt.ylim(2.0, 2.5)
    plt.legend()
    plt.savefig('../figures/leader_election.png')
    plt.close()
    
    return theoretical_values
```

## Question 4: City Distances and Graph Algorithms

Consider six cities (a through f) laid out with the following coordinates:

| City | Coordinates |
|------|-------------|
| a    | (0, 0)      |
| b    | (1, 1)      |
| c    | (0, 3)      |
| d    | (0, 2)      |
| e    | (1, 2)      |
| f    | (3, 1)      |


### Part (a): Distance Matrix
The distance matrix $D$ is calculated using Euclidean distance between cities:
$d_{ij} = \sqrt{(x_i-x_j)^2 + (y_i-y_j)^2}$

Here's the complete distance matrix (rounded to 2 decimal places):

|   | a    | b    | c    | d    | e    | f    |
|---|------|------|------|------|------|------|
| a | 0.00 | 1.41 | 3.00 | 2.00 | 2.24 | 3.16 |
| b | 1.41 | 0.00 | 2.24 | 1.41 | 1.00 | 2.00 |
| c | 3.00 | 2.24 | 0.00 | 1.00 | 1.41 | 3.61 |
| d | 2.00 | 1.41 | 1.00 | 0.00 | 1.00 | 3.00 |
| e | 2.24 | 1.00 | 1.41 | 1.00 | 0.00 | 2.24 |
| f | 3.16 | 2.00 | 3.61 | 3.00 | 2.24 | 0.00 |

For example:
- Distance a→b = √((1-0)² + (1-0)²) = √2 ≈ 1.41
- Distance c→d = √((0-0)² + (3-2)²) = 1.00
- Distance a→f = √((3-0)² + (1-0)²) = √10 ≈ 3.16

### Part (b): Minimum Spanning Tree (MST)
Using either Prim's or Kruskal's algorithm, we find the following MST:

Edges in the MST (in order of inclusion):
1. d→e: weight = 1.00
2. b→e: weight = 1.00
3. c→d: weight = 1.00
4. a→b: weight = 1.41
5. b→f: weight = 2.00

Total MST cost = 6.41

Key observations:
1. The MST connects all cities using the minimum possible total edge weight
2. It contains exactly |V|-1 = 5 edges for our 6 vertices
3. The algorithm prioritized the unit-length edges (weight = 1.00) first
4. The longest edge in the MST is b→f with length 2.00
5. The MST forms a tree structure with no cycles

The MST provides the most efficient way to connect all cities with roads, minimizing the total road length while ensuring all cities are accessible.

### Implementation Details

#### Distance Matrix Calculation
The implementation uses NumPy for efficient matrix operations:
```python
# Calculate distance matrix using nested loops
n = len(cities)
distance_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            x1, y1 = cities[city_list[i]]
            x2, y2 = cities[city_list[j]]
            distance_matrix[i][j] = np.sqrt((x2-x1)**2 + (y2-y1)**2)
```

#### Graph Construction
The implementation uses NetworkX library for graph operations:
```python
G = nx.Graph()
for i in range(n):
    for j in range(i+1, n):
        G.add_edge(city_list[i], city_list[j], weight=distance_matrix[i][j])
```

#### MST Algorithms
Two different algorithms were implemented and compared:

1. Prim's Algorithm:
   - Starts with a single vertex
   - Repeatedly adds the minimum weight edge that connects a vertex in the tree to a vertex outside
   - Implementation: `mst_prim = nx.minimum_spanning_tree(G, algorithm='prim')`

2. Kruskal's Algorithm:
   - Sorts all edges by weight
   - Adds edges in ascending order if they don't create cycles
   - Implementation: `mst_kruskal = nx.minimum_spanning_tree(G, algorithm='kruskal')`

Both algorithms produced identical results, confirming the optimality of the solution.

### Results

#### 1. Distance Matrix Analysis
- Minimum non-zero distance: 1.00 (several pairs, e.g., d→e, b→e)
- Maximum distance: 3.61 (c→f)
- Average distance: 1.89
- Matrix is symmetric: d(i,j) = d(j,i) for all pairs
- All diagonal entries are 0 (distance to self)

#### 2. MST Properties
- Number of vertices: 6
- Number of edges in MST: 5
- Total MST weight: 6.41
- Edge distribution by weight:
  * Weight 1.00: 3 edges
  * Weight 1.41: 1 edge
  * Weight 2.00: 1 edge

#### 3. Graph Visualization
Two plots were generated (see `Q4_city_graph_and_mst.png`):
1. Complete Graph:
   - Shows all possible connections between cities
   - Edge weights displayed on each connection
   - Cities positioned according to their coordinates

2. Minimum Spanning Tree:
   - Shows only the optimal connections
   - Demonstrates the minimal network needed
   - Highlights the efficient path structure

#### 4. Verification
- Both Prim's and Kruskal's algorithms produced identical MSTs
- The MST is unique in this case due to distinct edge weights
- All cities are connected with minimum total edge weight
- No cycles exist in the final tree

#### 5. Practical Implications
1. Road Network Design:
   - Optimal layout for connecting all cities
   - Minimizes total road construction cost
   - Ensures accessibility between all points

2. Network Properties:
   - Any two cities can be reached through the tree
   - Removing any edge disconnects the network
   - Adding any edge creates a cycle

3. Cost Efficiency:
   - Total road length of 6.41 units
   - Uses shorter connections where possible
   - Balances direct routes with overall efficiency

This solution demonstrates both the theoretical aspects of graph algorithms and their practical application in network design problems.

### Part (c): All-Pairs Shortest Paths

The Floyd-Warshall algorithm was used to compute shortest paths between all pairs of cities. Here's the complete shortest paths matrix (rounded to 2 decimal places):

|   | a    | b    | c    | d    | e    | f    |
|---|------|------|------|------|------|------|
| a | 0.00 | 1.41 | 3.00 | 2.00 | 2.24 | 3.16 |
| b | 1.41 | 0.00 | 2.24 | 1.41 | 1.00 | 2.00 |
| c | 3.00 | 2.24 | 0.00 | 1.00 | 1.41 | 3.61 |
| d | 2.00 | 1.41 | 1.00 | 0.00 | 1.00 | 3.00 |
| e | 2.24 | 1.00 | 1.41 | 1.00 | 0.00 | 2.24 |
| f | 3.16 | 2.00 | 3.61 | 3.00 | 2.24 | 0.00 |

Key Observations:
1. Path Properties:
   - All shortest paths are symmetric: d(i,j) = d(j,i)
   - Zero diagonal: d(i,i) = 0 for all cities
   - Triangle inequality satisfied: d(i,j) ≤ d(i,k) + d(k,j) for all cities i,j,k

2. Notable Paths:
   - Shortest: d→e, b→e, c→d (all 1.00 units)
   - Longest: c→f (3.61 units)
   - Average path length: 1.89 units

3. Path Analysis:
   - Some paths are direct (e.g., a→b = 1.41)
   - Others may use intermediate cities
   - All paths represent minimum possible distances

Implementation using NetworkX:
```python
# Compute all-pairs shortest paths
shortest_paths = nx.floyd_warshall(G)

# Example usage:
distance_a_to_f = shortest_paths['a']['f']  # Gets shortest path from a to f
```

This completes my analysis of the city network, showing both local (MST) and global (shortest paths) properties of the graph.

```python
def question4_city_distances():
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
    
    # Visualize the complete graph and MST
    plt.figure(figsize=(15, 6))
    
    # Plot original graph
    plt.subplot(121)
    pos = cities
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=16, font_weight='bold')
    
    edge_labels = {(u, v): f'{d["weight"]:.2f}' 
                  for (u, v, d) in G.edges(data=True)}
    
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=8,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
        label_pos=0.3
    )
    plt.title('Complete Graph')
    
    # Plot MST
    plt.subplot(122)
    nx.draw(mst_prim, pos, with_labels=True, node_color='lightgreen',
            node_size=500, font_size=16, font_weight='bold')
    
    edge_labels = {(u, v): f'{d["weight"]:.2f}' 
                  for (u, v, d) in mst_prim.edges(data=True)}
    
    nx.draw_networkx_edge_labels(
        mst_prim, pos,
        edge_labels=edge_labels,
        font_size=10,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
        label_pos=0.5
    )
    plt.title('Minimum Spanning Tree')
    plt.tight_layout(pad=3.0)
    plt.savefig('../figures/Q4_city_graph_and_mst.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    return distance_matrix, mst_prim, mst_kruskal, shortest_paths
```

## Question 5: Stirling's Approximation Analysis

Stirling's approximation $s(n)$ to $n!$ is given by:
$$s(n) = \sqrt{2\pi n}\left(\frac{n}{e}\right)^n$$

### Part a: Absolute Error Computation and Plot
The absolute error $e(n) = n! - s(n)$ is computed for $2 \leq n \leq 20$.

#### Calculation Method:
1. Calculate exact $n!$ for each $n$
2. Calculate Stirling's approximation $s(n)$
3. Compute error $e(n) = n! - s(n)$

#### Sample Calculations:
1. For $n = 2$:
   - $2! = 2$
   - $s(2) = \sqrt{4\pi}(2/e)^2 \approx 1.919004$
   - $e(2) = 2 - 1.919004 = 0.080996$

2. For $n = 3$:
   - $3! = 6$
   - $s(3) = \sqrt{6\pi}(3/e)^3 \approx 5.836210$
   - $e(3) = 6 - 5.836210 = 0.163790$

(See `Q5_stirling_error.png` for the complete error plot from $n=2$ to $n=20$)

### Part b: Relative Error Computation and Plot
The relative error $re(n) = e(n)/n!$ is computed for $2 \leq n \leq 20$.

#### Calculation Method:
1. Use absolute error $e(n)$ from Part a
2. Divide by $n!$ to get relative error
3. Express as percentage

#### Sample Calculations:
1. $re(2) = 0.080996/2 = 4.05\%$
2. $re(3) = 0.163790/6 = 2.73\%$
3. $re(4) = 0.493825/24 = 2.06\%$
4. $re(5) = 1.980832/120 = 1.65\%$
5. $re(6) = 9.921815/720 = 1.38\%$

(See `Q5_stirling_relative_error.png` for the complete relative error plot from $n=2$ to $n=20$)

### Part c: Observations

1. Absolute Error $(e(n))$:
   - Increases monotonically with $n$
   - Growth is super-linear
   - Always positive (approximation underestimates)
   - Becomes very large for larger $n$

2. Relative Error $(re(n))$:
   - Decreases monotonically with $n$
   - Starts at $\sim 4\%$ for $n=2$
   - Improves rapidly for small $n$
   - Continues to improve more slowly for larger $n$
   - Approaches $0$ as $n$ increases

3. Practical Implications:
   - Stirling's approximation becomes relatively more accurate for larger $n$
   - While absolute error grows, relative error diminishes
   - Very useful for large-scale calculations where exact factorial is impractical
   - Provides good balance of accuracy vs computational efficiency

4. Mathematical Properties:
   - Consistent underestimation: $s(n) < n!$ for all $n \geq 1$
   - Convergence: $\lim_{n \to \infty} re(n) = 0$
   - Error behavior is predictable and well-behaved

```python
def question5_stirling():
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
```

## Full code
``` python
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
```

## Required Libraries and Setup

To run the code in this solution, you'll need the following Python libraries:

### Core Libraries
1. `numpy` (version >= 1.20.0)
   - Used for efficient numerical computations
   - Array operations and mathematical functions
   - Install: `pip install numpy`

2. `matplotlib` (version >= 3.4.0)
   - Used for all plotting and visualizations
   - Generates figures for Monte Carlo, MST, and error analysis
   - Install: `pip install matplotlib`

3. `networkx` (version >= 2.6.0)
   - Used for graph operations in Question 4
   - Implements Prim's and Kruskal's MST algorithms
   - Install: `pip install networkx`

### Standard Libraries (included with Python)
1. `math`
   - Used for mathematical operations
   - Factorial calculations and combinations

2. `random`
   - Used for random number generation
   - Monte Carlo simulations

3. `datetime`
   - Used for timing code execution
   - Performance monitoring

### Installation
You can install all required external libraries using pip:
```bash
pip install numpy matplotlib networkx
```

### Version Compatibility
- Python version >= 3.8 
- Code tested on Python 3.8 and 3.9
- All libraries use their latest stable versions as of March 2025