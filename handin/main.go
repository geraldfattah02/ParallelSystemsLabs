// main.go
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"handin/bst"
	"handin/buffer"
	"handin/hash"
	"handin/utils"
)

func main() {
	// --- flags ---
	hashWorkers := flag.Int("hash-workers", 1, "number of hashing worker goroutines (0 => one goroutine per tree)")
	dataMode := flag.String("data-mode", "channel", "how to collect hashes: 'channel' or 'lock'")
	compWorkers := flag.Int("comp-workers", 4, "number of comparison worker goroutines (used in comp-mode=pool)")
	compMode := flag.String("comp-mode", "goroutine", "comparison mode: 'goroutine' or 'pool'")
	input := flag.String("input", "simple.txt", "input file path")
	flag.Parse()

	startTotal := time.Now()

	// --- Step 1: Read & build trees ---
	t0 := time.Now()
	lines, err := utils.ReadLines(*input)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error reading input:", err)
		os.Exit(1)
	}
	trees := make([]*bst.BST, 0, len(lines))
	for _, line := range lines {
		if strings.TrimSpace(line) == "" {
			continue
		}
		nums := utils.ParseLine(line)
		t := &bst.BST{}
		for _, v := range nums {
			t.Insert(v)
		}
		trees = append(trees, t)
	}
	fmt.Printf("Read & built %d trees in %v\n", len(trees), time.Since(t0))

	// Preallocate slices
	inOrders := make([][]int, len(trees))
	hashes := make([]string, len(trees))

	// --- Step 2: Hashing ---
	t1 := time.Now()
	if *hashWorkers == 0 {
		var wg sync.WaitGroup
		wg.Add(len(trees))
		for i := range trees {
			go func(idx int) {
				defer wg.Done()
				io := trees[idx].InOrderSlice()
				inOrders[idx] = io
				hashes[idx] = hash.HashInts(io)
			}(i)
		}
		wg.Wait()
	} else {
		jobs := make(chan int)
		var wg sync.WaitGroup
		for w := 0; w < *hashWorkers; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for idx := range jobs {
					io := trees[idx].InOrderSlice()
					inOrders[idx] = io
					hashes[idx] = hash.HashInts(io)
				}
			}()
		}
		for i := range trees {
			jobs <- i
		}
		close(jobs)
		wg.Wait()
	}
	fmt.Printf("Hashed %d trees in %v\n", len(trees), time.Since(t1))

	// --- Step 3: Build hash map ---
	t2 := time.Now()
	hashToIDs := make(map[string][]int)
	if *dataMode == "channel" {
		ch := make(chan struct {
			h string
			i int
		})
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			for p := range ch {
				hashToIDs[p.h] = append(hashToIDs[p.h], p.i)
			}
		}()
		for i, h := range hashes {
			ch <- struct {
				h string
				i int
			}{h: h, i: i}
		}
		close(ch)
		wg.Wait()
	} else {
		var mu sync.Mutex
		var wg sync.WaitGroup
		for i, h := range hashes {
			wg.Add(1)
			go func(idx int, hh string) {
				defer wg.Done()
				mu.Lock()
				hashToIDs[hh] = append(hashToIDs[hh], idx)
				mu.Unlock()
			}(i, h)
		}
		wg.Wait()
	}
	fmt.Printf("Built hash map in %v\n", time.Since(t2))

	// --- Step 4: Comparisons ---
	t3 := time.Now()
	n := len(trees)
	adj := utils.NewAdjMatrix(n)

	// Helper comparator
	areEqual := func(a, b int) bool {
		ia := inOrders[a]
		ib := inOrders[b]
		if len(ia) != len(ib) {
			return false
		}
		for i := range ia {
			if ia[i] != ib[i] {
				return false
			}
		}
		return true
	}

	// Build work pairs
	workPairs := utils.BuildWorkPairs(hashToIDs, adj)

	if *compMode == "goroutine" {
		var wg sync.WaitGroup
		for _, p := range workPairs {
			wg.Add(1)
			go func(a, b int) {
				defer wg.Done()
				if areEqual(a, b) {
					adj[a][b] = true
					adj[b][a] = true
				}
			}(p[0], p[1])
		}
		wg.Wait()
	} else {
		buf := buffer.NewBoundedBuffer(*compWorkers)
		var workers sync.WaitGroup
		for w := 0; w < *compWorkers; w++ {
			workers.Add(1)
			go func() {
				defer workers.Done()
				for {
					item, ok := buf.Pop()
					if !ok {
						return
					}
					a, b := item[0], item[1]
					if areEqual(a, b) {
						adj[a][b] = true
						adj[b][a] = true
					}
				}
			}()
		}
		for _, p := range workPairs {
			if !buf.Push(p) {
				break
			}
		}
		buf.Close()
		workers.Wait()
	}

	fmt.Printf("Compared %d pairs in %v\n", len(workPairs), time.Since(t3))

	// --- Step 5: Group detection ---
	groups := utils.FindGroupsFromAdj(adj)
	fmt.Printf("Found %d groups of equivalent trees\n", len(groups))
	for gi, g := range groups {
		fmt.Printf("group %d: %v\n", gi, g)
	}

	fmt.Printf("Total elapsed: %v\n", time.Since(startTotal))
}
