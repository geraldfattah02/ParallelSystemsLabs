package hash

import (
	"bytes"
	"crypto/sha1"
	"fmt"
	"strconv"
)

func HashInts(a []int) string {
	var b bytes.Buffer
	for i, v := range a {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.Itoa(v))
	}
	h := sha1.Sum([]byte(b.String()))
	return fmt.Sprintf("%x", h)
}
