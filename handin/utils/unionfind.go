package utils

type UnionFind struct {
	parent []int
	rank   []int
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
	if uf.parent[x] != x {
		uf.parent[x] = uf.Find(uf.parent[x])
	}
	return uf.parent[x]
}

func (uf *UnionFind) Union(a, b int) {
	ra, rb := uf.Find(a), uf.Find(b)
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

func (uf *UnionFind) Groups() [][]int {
	rootToGroup := make(map[int][]int)
	for i := range uf.parent {
		root := uf.Find(i)
		rootToGroup[root] = append(rootToGroup[root], i)
	}
	groups := make([][]int, 0, len(rootToGroup))
	for _, g := range rootToGroup {
		groups = append(groups, g)
	}
	return groups
}
