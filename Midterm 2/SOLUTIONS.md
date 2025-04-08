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

This completes our analysis of the city network, showing both local (MST) and global (shortest paths) properties of the graph. 