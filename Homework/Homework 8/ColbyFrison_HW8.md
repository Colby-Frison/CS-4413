# Homework 8
**Name**: Colby Frison
**OUID**: 1135568816
**Date**: 4/7/2025
**Class**: CS-4413

# Questions

## 1. Problem 22.1-7 ~ Incidence Matrix and $BB^T$ interpretation

### Incidence matrix

In analyzing the directed graph from Figure 22.2(a), we will examine the properties of its incidence matrix B and the resulting matrix BB^T. 

To begin our analysis, we must first construct the incidence matrix B. The graph contains 6 vertices and 8 edges in total, relationships are shown below:

- $e_{1} : 1 \rightarrow 2$
- $e_{2} : 1 \rightarrow 4$
- $e_{3} : 2 \rightarrow 5$
- $e_{4} : 3 \rightarrow 5$
- $e_{5} : 3 \rightarrow 6$
- $e_{6} : 4 \rightarrow 2$
- $e_{7} : 5 \rightarrow 4$

Total edges: 7

Each row(1-7) corresponds to a vertex, and each column(e1-e7) corresponds to an edge. The 6x7 matrix uses the following incidence rule:

Using the incidence rule:
 - $b_{ij} = -1$ if edge $j$ leaves vertex $i$
 - $b_{ij} = 1$ if edge $j$ enters vertex $i$
 - $b_{ij} = 0$ otherwise

    |   | e1 | e2 | e3 | e4 | e5 | e6 | e7 |
    |---|----|----|----|----|----|----|----|
    | 1 | -1 | -1 |  0 |  0 |  0 |  0 |  0 |
    | 2 |  1 |  0 | -1 |  0 |  0 |  1 |  0 |
    | 3 |  0 |  0 |  0 | -1 | -1 |  0 |  0 |
    | 4 |  0 |  1 |  0 |  0 |  0 | -1 |  1 |
    | 5 |  0 |  0 |  1 |  1 |  0 |  0 | -1 |
    | 6 |  0 |  0 |  0 |  0 |  1 |  0 |  0 |

### What does $BB^T$ represent

Let $A = BB^T$ where B is our incidence matrix. The resulting matrix A provides crucial information about the graph's structure. For any entry $A_{ij}$ in A:

For diagonal entries $(i = j)$:
- $A_{ii}$ equals the total degree of vertex i (both in-degree and out-degree)
- This is because when we multiply row i of B with column i of $B^T$, each edge incident to vertex i contributes exactly 1 to the sum:
  - For an outgoing edge: $(-1)^2 = 1$
  - For an incoming edge: $(1)^2 = 1$

For off-diagonal entries $(i \neq j)$:
- $A_{ij}$ represents the relationship between vertices i and j
- When multiplying row i by column j:
  - If there's an edge from i to j: $(-1)(1) = -1$
  - If there's an edge from j to i: $(1)(-1) = -1$
  - If no edge exists between i and j: $0$

The fundamental insight is that $BB^T$ **creates a vertex-vertex matrix that reveals how vertices are connected through their shared edges, while preserving directional information.** While diagonal entries $A_{ii}$ give us vertex degrees, the off-diagonal entries $A_{ij}$ are the key feature, showing us the relationships between pairs of vertices based on their edge connections and directions.

--- 

## 2. Problem 22.2-8 ~ Diameter of a Graph

The diameter of a tree $T = (V,E)$ is defined as the maximum distance between any two vertices in the tree, where distance is measured by the number of edges in the shortest path. We can find this using an application of BFS. The key insight is that we don't need to check every possible pair of vertices - we can find the diameter with just two BFS traversals.

### Analysis

The solution works by using these steps:

1. Run first BFS:
   - Pick any starting vertex $u$
   - Run BFS from $u$ to find the furthest vertex $v$
   - The vertex $v$ will be one end of the diameter

2. Run second BFS:
   - Start from vertex $v$ that we found
   - Run BFS again to find the furthest vertex $w$
   - The path from $v$ to $w$ gives us the diameter

This works because in a tree, the diameter must be a path between two leaf nodes. When we run the first BFS, we're guaranteed to find one end of the diameter path. The second BFS then finds the other end.

### Time Complexity

The algorithm is efficient because:
- Each BFS only needs to visit each vertex once
- In a tree, $|E| = |V| - 1$ (number of edges is one less than vertices)
- We do two BFS traversals, each taking $O(V)$ time
- Total runtime: $O(V)$

This is optimal since we need to at least look at each vertex once to find the diameter.

---

## 3. Shortest path form a to p

### Using BFS
Breadth-First Search (BFS) is ideal for finding the shortest path in an unweighted graph (or a graph where all edge weights are the same, as in this case). BFS explores vertices layer by layer, ensuring that the first time we reach a vertex, we have found the shortest path to it in terms of the number of edges.

Steps for BFS:
    1. Start at vertex aa.
    2. Use a queue to explore vertices in order of distance (number of edges) from aa.
    3. Keep track of the parent of each vertex to reconstruct the path.
    4. Stop when we reach vertex pp.

BFS Execution:
- **Queue**: Start with vertex aa
- **Visited**: Mark vertex aa as visited
- **Distances**: Initialize distance for vertex *a* as 0
- **Parents**: Track the parent of each vertex for path reconstruction

**Step 1**: Dequeue $a$, explore neighbors $b$ and $c$.
- Enqueue $b$ and $c$
- Distance: $b:1$, $c:1$
- Parents: $b \leftarrow a$, $c \leftarrow a$
- Queue: $b,c$

**Step 2**: Dequeue $b$, explore neighbors $d,e$ (skip $a$)
- Enqueue $d$ and $e$
- Distance: $d:2$, $e:2$
- Parents: $d \leftarrow b$, $e \leftarrow b$
- Queue: $c,d,e$

**Step 3**: Dequeue $c$, explore neighbors $e,f$ (skip $a$)
- $e$ is already visited, so enqueue $f$
- Distance: $f:2$
- Parent: $f \leftarrow c$
- Queue: $d,e,f$

**Step 4**: Dequeue $d$, explore neighbors $g,h$ (skip $b$)
- Enqueue $g$ and $h$
- Distance: $g:3$, $h:3$
- Parents: $g \leftarrow d$, $h \leftarrow d$
- Queue: $e,f,g,h$

**Step 5**: Dequeue $e$, explore neighbors $h,i$ (skip $b,c$)
- $h$ is already visited, so enqueue $i$
- Distance: $i:3$
- Parent: $i \leftarrow e$
- Queue: $f,g,h,i$

**Step 6**: Dequeue $f$, explore neighbors $i,j$ (skip $c$)
- $i$ is already visited, so enqueue $j$
- Distance: $j:3$
- Parent: $j \leftarrow f$
- Queue: $g,h,i,j$

**Step 7**: Dequeue $g$, explore neighbor $k$ (skip $d$)
- Enqueue $k$
- Distance: $k:4$
- Parent: $k \leftarrow g$
- Queue: $h,i,j,k$

**Step 8**: Dequeue $h$, explore neighbors $k,l$ (skip $d,e$)
- $k$ is already visited, so enqueue $l$
- Distance: $l:4$
- Parent: $l \leftarrow h$
- Queue: $i,j,k,l$

**Step 9**: Dequeue $i$, explore neighbors $l,m$ (skip $e,f$)
- $l$ is already visited, so enqueue $m$
- Distance: $m:4$
- Parent: $m \leftarrow i$
- Queue: $j,k,l,m$

**Step 10**: Dequeue $j$, explore neighbor $m$ (skip $f$)
- $m$ is already visited
- Queue: $k,l,m$

**Step 11**: Dequeue $k$, explore neighbor $n$ (skip $g,h$)
- Enqueue $n$
- Distance: $n:5$
- Parent: $n \leftarrow k$
- Queue: $l,m,n$

**Step 12**: Dequeue $l$, explore neighbors $n,o$ (skip $h,i$)
- $n$ is already visited, so enqueue $o$
- Distance: $o:5$
- Parent: $o \leftarrow l$
- Queue: $m,n,o$

**Step 13**: Dequeue $m$, explore neighbor $o$ (skip $i,j$)
- $o$ is already visited
- Queue: $n,o$

**Step 14**: Dequeue $n$, explore neighbor $p$ (skip $k,l$)
- Enqueue $p$
- Distance: $p:6$
- Parent: $p \leftarrow n$
- Queue: $o,p$

**Step 15**: Dequeue $o$, explore neighbor $p$ (skip $l,m$)
- $p$ is already visited
- Queue: $p$

**Step 16**: Dequeue $p$, we've reached the target

Path Reconstruction:
- $p \leftarrow n \leftarrow k \leftarrow g \leftarrow d \leftarrow b \leftarrow a$
- Path: $a \rightarrow b \rightarrow d \rightarrow g \rightarrow k \rightarrow n \rightarrow p$
- Path length: 6 edges

Shortest Path using BFS:

The shortest path from $a$ to $p$ is $a \rightarrow b \rightarrow d \rightarrow g \rightarrow k \rightarrow n \rightarrow p$, with a total distance of 6 (since each edge has weight 1).

### Using Dijkstra's algorithm

Dijkstra's algorithm is typically used for graphs with non-negative edge weights. Since all edge weights are 1, Dijkstra's algorithm will behave similarly to BFS in this case, as it will explore vertices in order of increasing distance from the source.

Steps for Dijkstra's Algorithm:
1. Start at vertex $a$
2. Maintain a priority queue of vertices, prioritized by their distance from $a$
3. Keep track of parents for path reconstruction
4. Stop when we reach vertex $p$

Dijkstra's Execution:
- **Priority Queue**: Start with vertex $a$
- **Visited**: Empty initially
- **Distances**: Initialize $a:0$, all others: $\infty$
- **Parents**: Track parent of each vertex for path reconstruction

**Step 1**: Extract $a$ (distance 0), explore neighbors $b$ and $c$
- Update distances: $b:1$, $c:1$
- Parents: $b \leftarrow a$, $c \leftarrow a$
- Queue: $(b,1),(c,1)$

**Step 2**: Extract $b$ (distance 1), explore neighbors $d,e$
- Update distances: $d:2$, $e:2$
- Parents: $d \leftarrow b$, $e \leftarrow b$
- Queue: $(c,1),(d,2),(e,2)$

**Step 3**: Extract $c$ (distance 1), explore neighbors $e,f$
- Update: $e:2$ (no change), $f:2$
- Parent: $f \leftarrow c$
- Queue: $(d,2),(e,2),(f,2)$

**Step 4**: Extract $d$ (distance 2), explore neighbors $g,h$
- Update distances: $g:3$, $h:3$
- Parents: $g \leftarrow d$, $h \leftarrow d$
- Queue: $(e,2),(f,2),(g,3),(h,3)$

**Step 5**: Extract $e$ (distance 2), explore neighbors $h,i$
- Update: $h:3$ (no change), $i:3$
- Parent: $i \leftarrow e$
- Queue: $(f,2),(g,3),(h,3),(i,3)$

**Step 6**: Extract $f$ (distance 2), explore neighbors $i,j$
- Update: $i:3$ (no change), $j:3$
- Parent: $j \leftarrow f$
- Queue: $(g,3),(h,3),(i,3),(j,3)$

**Step 7**: Extract $g$ (distance 3), explore neighbor $k$
- Update distance: $k:4$
- Parent: $k \leftarrow g$
- Queue: $(h,3),(i,3),(j,3),(k,4)$

**Step 8**: Extract $h$ (distance 3), explore neighbors $k,l$
- Update: $k:4$ (no change), $l:4$
- Parent: $l \leftarrow h$
- Queue: $(i,3),(j,3),(k,4),(l,4)$

**Step 9**: Extract $i$ (distance 3), explore neighbors $l,m$
- Update: $l:4$ (no change), $m:4$
- Parent: $m \leftarrow i$
- Queue: $(j,3),(k,4),(l,4),(m,4)$

**Step 10**: Extract $j$ (distance 3), explore neighbor $m$
- Update: $m:4$ (no change)
- Queue: $(k,4),(l,4),(m,4)$

**Step 11**: Extract $k$ (distance 4), explore neighbor $n$
- Update distance: $n:5$
- Parent: $n \leftarrow k$
- Queue: $(l,4),(m,4),(n,5)$

**Step 12**: Extract $l$ (distance 4), explore neighbors $n,o$
- Update: $n:5$ (no change), $o:5$
- Parent: $o \leftarrow l$
- Queue: $(m,4),(n,5),(o,5)$

**Step 13**: Extract $m$ (distance 4), explore neighbor $o$
- Update: $o:5$ (no change)
- Queue: $(n,5),(o,5)$

**Step 14**: Extract $n$ (distance 5), explore neighbor $p$
- Update distance: $p:6$
- Parent: $p \leftarrow n$
- Queue: $(o,5),(p,6)$

**Step 15**: Extract $o$ (distance 5), explore neighbor $p$
- Update: $p:6$ (no change)
- Queue: $(p,6)$

**Step 16**: Extract $p$ (distance 6), we've reached the target

Path Reconstruction:
- $p \leftarrow n \leftarrow k \leftarrow g \leftarrow d \leftarrow b \leftarrow a$
- Path: $a \rightarrow b \rightarrow d \rightarrow g \rightarrow k \rightarrow n \rightarrow p$
- Path length: 6 edges

### Conclusion

Both BFS and Dijkstra's algorithm found the same shortest path from $a$ to $p$:
- Path: $a \rightarrow b \rightarrow d \rightarrow g \rightarrow k \rightarrow n \rightarrow p$
- Distance: 6 edges

This is expected since all edge weights are 1, making both algorithms behave similarly. Note that there are other paths of the same length (e.g., $a \rightarrow c \rightarrow f \rightarrow j \rightarrow m \rightarrow o \rightarrow p$), but either path is a valid shortest path.