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
	// command-line flags
	hashWorkers := flag.Int("hash-workers", 1, "number of hashing goroutines (0 = one per tree)")
	dataWorkers := flag.Int("data-workers", 1, "number of workers to update the map")
	compWorkers := flag.Int("comp-workers", 1, "number of goroutines used for tree comparison")
	compMode := flag.String("comp-mode", "goroutine", "comparison mode: 'goroutine' or 'pool'")
	input := flag.String("input", "simple.txt", "path to input file")
	flag.Parse()

	// read all trees from input
	lines, err := utils.ReadLines(*input)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error reading input:", err)
		os.Exit(1)
	}

	var trees []*bst.BST
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		nums := utils.ParseLine(line)
		t := &bst.BST{}
		for _, v := range nums {
			t.Insert(v)
		}
		trees = append(trees, t)
	}

	n := len(trees)
	inOrders := make([][]int, n)
	hashes := make([]string, n)

	// --- compute hashes ---
	tStart := time.Now()

	if *hashWorkers == 0 {
		var wg sync.WaitGroup
		wg.Add(n)
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

	fmt.Printf("hashTime: %.6f\n", time.Since(tStart).Seconds())

	// --- build hash groups ---
	hashToIDs := make(map[string][]int)

	switch {
	case *hashWorkers == 1 && *dataWorkers == 1:
		// sequential
		for i, h := range hashes {
			hashToIDs[h] = append(hashToIDs[h], i)
		}

	case *dataWorkers == 1:
		// one goroutine manages all map updates via a channel
		ch := make(chan struct {
			h string
			i int
		}, len(hashes))
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
			}{h, i}
		}
		close(ch)
		wg.Wait()

	case *dataWorkers == *hashWorkers:
		// multiple workers update the map with a lock
		var mu sync.Mutex
		var wg sync.WaitGroup
		type pair struct {
			h string
			i int
		}
		jobs := make(chan pair, len(hashes))

		for w := 0; w < *dataWorkers; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for p := range jobs {
					mu.Lock()
					hashToIDs[p.h] = append(hashToIDs[p.h], p.i)
					mu.Unlock()
				}
			}()
		}
		for i, h := range hashes {
			jobs <- pair{h: h, i: i}
		}
		close(jobs)

		wg.Wait()

	default:
		fmt.Fprintf(os.Stderr, "unsupported flag combo: hash-workers=%d data-workers=%d\n", *hashWorkers, *dataWorkers)
		os.Exit(1)
	}

	fmt.Printf("hashGroupTime: %.6f\n", time.Since(tStart).Seconds())

	// print only groups with more than one element
	groupCount := 0
	for _, ids := range hashToIDs {
		if len(ids) > 1 {
			fmt.Printf("hash%d:", groupCount)
			for _, id := range ids {
				fmt.Printf(" %d", id)
			}
			fmt.Println()
			groupCount++
		}
	}

	// --- tree comparison ---
	tCompare := time.Now()
	uf := utils.NewUnionFind(n)

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

	workPairs := utils.BuildWorkPairs(hashToIDs)

	if *compMode == "goroutine" {
		var wg sync.WaitGroup
		var mu sync.Mutex

		for _, p := range workPairs {
			wg.Add(1)
			go func(a, b int) {
				defer wg.Done()
				if areEqual(a, b) {
					mu.Lock()
					uf.Union(a, b)
					mu.Unlock()
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
						uf.Union(a, b)
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

	fmt.Printf("compareTreeTime: %.6f\n", time.Since(tCompare).Seconds())

	// --- print final tree groups ---
	finalGroups := uf.Groups()
	idx := 0
	for _, g := range finalGroups {
		if len(g) > 1 {
			fmt.Printf("group %d:", idx)
			for _, id := range g {
				fmt.Printf(" %d", id)
			}
			fmt.Println()
			idx++
		}
	}
}
