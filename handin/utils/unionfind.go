package utils

import "sync"

type UnionFind struct {
	parent []int
	rank   []int
	mu     sync.Mutex
}

func NewUnionFind(n int) *UnionFind {
	p := make([]int, n)
	r := make([]int, n)
	for i := range p {
		p[i] = i
	}
	return &UnionFind{parent: p, rank: r}
}

func (uf *UnionFind) Find(x int) int {
	// Iterative path compression
	uf.mu.Lock()
	defer uf.mu.Unlock()
	root := x
	for uf.parent[root] != root {
		root = uf.parent[root]
	}
	// Path compression
	for x != root {
		parent := uf.parent[x]
		uf.parent[x] = root
		x = parent
	}
	return root
}
func (uf *UnionFind) Union(a, b int) {
	uf.mu.Lock()
	defer uf.mu.Unlock()

	ra, rb := uf.findNoLock(a), uf.findNoLock(b)
	if ra == rb {
		return
	}
	if uf.rank[ra] < uf.rank[rb] {
		uf.parent[ra] = rb
	} else if uf.rank[ra] > uf.rank[rb] {
		uf.parent[rb] = ra
	} else {
		uf.parent[rb] = ra
		uf.rank[ra]++
	}
}

func (uf *UnionFind) findNoLock(x int) int {
	root := x
	for uf.parent[root] != root {
		root = uf.parent[root]
	}
	for x != root {
		parent := uf.parent[x]
		uf.parent[x] = root
		x = parent
	}
	return root
}

func (uf *UnionFind) Groups() [][]int {
	uf.mu.Lock()
	defer uf.mu.Unlock()

	rootToGroup := make(map[int][]int)
	for i := range uf.parent {
		root := uf.findNoLock(i)
		rootToGroup[root] = append(rootToGroup[root], i)
	}
	groups := make([][]int, 0, len(rootToGroup))
	for _, g := range rootToGroup {
		groups = append(groups, g)
	}
	return groups
}
