package utils

import (
	"bufio"
	"os"
	"strconv"
	"strings"
)

func ReadLines(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	out := []string{}
	for scanner.Scan() {
		out = append(out, scanner.Text())
	}
	return out, scanner.Err()
}

func ParseLine(line string) []int {
	parts := strings.Fields(line)
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		v, err := strconv.Atoi(p)
		if err == nil {
			out = append(out, v)
		}
	}
	return out
}

func NewAdjMatrix(n int) [][]bool {
	adj := make([][]bool, n)
	for i := range adj {
		adj[i] = make([]bool, n)
	}
	return adj
}

func BuildWorkPairs(hashToIDs map[string][]int, adj [][]bool) [][2]int {
	work := make([][2]int, 0)
	for _, ids := range hashToIDs {
		if len(ids) == 1 {
			k := ids[0]
			adj[k][k] = true
			continue
		}
		for i := 0; i < len(ids); i++ {
			for j := i; j < len(ids); j++ {
				work = append(work, [2]int{ids[i], ids[j]})
			}
		}
	}
	return work
}

func FindGroupsFromAdj(adj [][]bool) [][]int {
	n := len(adj)
	visited := make([]bool, n)
	groups := [][]int{}
	for i := 0; i < n; i++ {
		if visited[i] {
			continue
		}
		stack := []int{i}
		group := []int{}
		visited[i] = true
		for len(stack) > 0 {
			u := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			group = append(group, u)
			for v := 0; v < n; v++ {
				if !visited[v] && adj[u][v] {
					visited[v] = true
					stack = append(stack, v)
				}
			}
		}
		groups = append(groups, group)
	}
	return groups
}
