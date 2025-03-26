# Algorithm Analysis - Spring 2025 Midterm 2 Solutions

## Question 1: Monte Carlo Method for π Estimation

### Part A: Generating Random Points
We generate N pairs of points (ui, vi) uniformly distributed in [0,1). These points are plotted on a unit square.

### Part B: Estimating π/4
The Monte Carlo method for estimating π uses the relationship between the area of a circle and its circumscribing square.

#### Mathematical Foundation:
- Area of unit square = 1 × 1 = 1
- Area of quarter circle = π/4
- Probability of point falling inside quarter circle = (π/4)/(1) = π/4

#### Implementation:
1. Generate random points (x,y) in [0,1) × [0,1)
2. Count points where x² + y² ≤ 1 (inside quarter circle)
3. Estimate: π/4 ≈ (points inside)/(total points)

#### Results:
- N = 10³: π ≈ 3.208000 (Error: 2.17%)
- N = 10⁴: π ≈ 3.131640 (Error: 0.32%)
- N = 10⁵: π ≈ 3.139152 (Error: 0.08%)
- N = 10⁶: π ≈ 3.141593 (Error: 0.0001%)

The error decreases as N increases, demonstrating the Law of Large Numbers.

## Question 2: Parenthesizing Expressions

### Recurrence Relation
Let P(n) be the number of ways to parenthesize n atoms.

#### Derivation:
For n atoms, we can split the expression after k atoms (1 ≤ k < n):
P(n) = ∑(k=1 to n-1) P(k) × P(n-k)

#### Base Cases:
- P(1) = 1 (single atom needs no parentheses)
- P(2) = 1 (only one way to parenthesize two atoms)

#### First Few Values:
- P(2) = 1
- P(3) = P(1)×P(2) + P(2)×P(1) = 2
- P(4) = P(1)×P(3) + P(2)×P(2) + P(3)×P(1) = 5
- P(5) = P(1)×P(4) + P(2)×P(3) + P(3)×P(2) + P(4)×P(1) = 14

These numbers are known as Catalan numbers.

## Question 3: Leader Election Rounds

### Recurrence Relation
Let L(n) be the average number of rounds needed for n people.

#### Derivation:
In each round:
- One person is selected with probability 1/n
- If not selected (probability (n-1)/n), we need L(n-1) more rounds

Therefore:
L(n) = 1 + (n-1)/n × L(n-1)

#### Base Cases:
- L(1) = 0 (no rounds needed for one person)
- L(2) = 1 (exactly one round needed for two people)

#### First Few Values:
- L(2) = 1.0000
- L(3) = 1 + (2/3)×1 = 1.6667
- L(4) = 1 + (3/4)×1.6667 = 2.2500
- L(5) = 1 + (4/5)×2.2500 = 2.8000
- L(6) = 1 + (5/6)×2.8000 = 3.3333

## Question 4: City Distances and Graph Algorithms

### Part A: Distance Matrix Computation

Given six cities (a through f) on a grid, we compute pairwise distances.

#### City Coordinates:
- a: (0,0)
- b: (1,0)
- c: (2,0)
- d: (0,1)
- e: (1,1)
- f: (2,1)

#### Distance Calculations:
Using Euclidean distance formula: d = √[(x₂-x₁)² + (y₂-y₁)²]

Example calculations:
1. d(a,b) = √[(1-0)² + (0-0)²] = 1
2. d(a,e) = √[(1-0)² + (1-0)²] = √2 ≈ 1.4142
3. d(a,f) = √[(2-0)² + (1-0)²] = √5 ≈ 2.2361

Complete Distance Matrix:
```
    a    b    c    d    e    f
a   0    1    2    1    1.41 2.24
b   1    0    1    1.41 1    1.41
c   2    1    0    2.24 1.41 1
d   1    1.41 2.24 0    1    2
e   1.41 1    1.41 1    0    1
f   2.24 1.41 1    2    1    0
```

### Part B: Minimum Spanning Tree (MST)

#### Using Prim's Algorithm:
1. Start from vertex a
2. Add edge (a,b) with weight 1
3. Add edge (b,e) with weight 1
4. Add edge (e,f) with weight 1
5. Add edge (b,c) with weight 1
6. Add edge (d,e) with weight 1

Total MST cost = 5.0

#### Using Kruskal's Algorithm:
1. Sort edges by weight:
   - (a,b), (b,c), (d,e), (e,f) all with weight 1
   - (b,e), (c,f), etc. with weight √2
2. Add edges in order (avoiding cycles)
   - Same result as Prim's algorithm

Total MST cost = 5.0

### Part C: All-Pairs Shortest Paths

Using Floyd-Warshall algorithm:
- Initial distances = direct distances from distance matrix
- For each intermediate vertex k:
  - For each pair (i,j):
    - Update d[i,j] = min(d[i,j], d[i,k] + d[k,j])

Final shortest path distances are the same as the original distance matrix because the graph is embedded in a 2D plane and all direct paths are optimal.

## Question 5: Stirling's Approximation Analysis

### Mathematical Foundation
Stirling's approximation for n!:
s(n) = √(2πn)(n/e)ⁿ

### Error Analysis
#### Absolute Error:
e(n) = n! - s(n)

First few values:
- e(2) = 2! - s(2) = 2 - 1.919004 = 0.080996
- e(3) = 6 - 5.836210 = 0.163790
- e(4) = 24 - 23.506175 = 0.493825
- e(5) = 120 - 118.019168 = 1.980832
- e(6) = 720 - 710.078185 = 9.921815

#### Relative Error:
re(n) = e(n)/n!

First few values:
- re(2) = 0.040498 (4.05%)
- re(3) = 0.027298 (2.73%)
- re(4) = 0.020576 (2.06%)
- re(5) = 0.016507 (1.65%)
- re(6) = 0.013780 (1.38%)

### Observations
1. The absolute error increases with n
2. The relative error decreases with n
3. Stirling's approximation becomes more accurate (relatively) for larger values of n
4. The approximation is always an underestimate of the true factorial 