package buffer

import "sync"

type BoundedBuffer struct {
	buf      [][2]int
	capacity int
	mutex    sync.Mutex
	notFull  *sync.Cond
	notEmpty *sync.Cond
	closed   bool
}

func NewBoundedBuffer(cap int) *BoundedBuffer {
	b := &BoundedBuffer{
		buf:      make([][2]int, 0, cap),
		capacity: cap,
	}
	b.notFull = sync.NewCond(&b.mutex)
	b.notEmpty = sync.NewCond(&b.mutex)
	return b
}

func (b *BoundedBuffer) Push(item [2]int) bool {
	b.mutex.Lock()
	defer b.mutex.Unlock()
	for len(b.buf) >= b.capacity && !b.closed {
		b.notFull.Wait()
	}
	if b.closed {
		return false
	}
	b.buf = append(b.buf, item)
	b.notEmpty.Signal()
	return true
}

func (b *BoundedBuffer) Pop() ([2]int, bool) {
	b.mutex.Lock()
	defer b.mutex.Unlock()
	for len(b.buf) == 0 && !b.closed {
		b.notEmpty.Wait()
	}
	if len(b.buf) == 0 {
		return [2]int{}, false
	}
	item := b.buf[0]
	b.buf = b.buf[1:]
	b.notFull.Signal()
	return item, true
}

func (b *BoundedBuffer) Close() {
	b.mutex.Lock()
	b.closed = true
	b.notFull.Broadcast()
	b.notEmpty.Broadcast()
	b.mutex.Unlock()
}
