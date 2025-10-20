package bst

type Node struct {
	Val   int
	Left  *Node
	Right *Node
}

type BST struct {
	Root *Node
}

func (t *BST) Insert(v int) {
	if t.Root == nil {
		t.Root = &Node{Val: v}
		return
	}
	cur := t.Root
	for {
		if v < cur.Val {
			if cur.Left == nil {
				cur.Left = &Node{Val: v}
				return
			}
			cur = cur.Left
		} else {
			if cur.Right == nil {
				cur.Right = &Node{Val: v}
				return
			}
			cur = cur.Right
		}
	}
}

func (t *BST) InOrderSlice() []int {
	out := []int{}
	var dfs func(n *Node)
	dfs = func(n *Node) {
		if n == nil {
			return
		}
		dfs(n.Left)
		out = append(out, n.Val)
		dfs(n.Right)
	}
	dfs(t.Root)
	return out
}
