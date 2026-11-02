// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"
	"os"
	"sync"
	"time"

	"golang.org/x/tools/gopls/internal/file"
	"golang.org/x/tools/gopls/internal/label"
	"golang.org/x/tools/gopls/internal/protocol"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/robustio"
)

// A memoizedFS is a file source that memoizes reads, to reduce IO.
type memoizedFS struct {
	mu sync.Mutex

	// filesByID maps existing file inodes to the result of a read.
	// (The read may have failed, e.g. due to EACCES or a delete between stat+read.)
	// Each slice is a non-empty list of aliases: different URIs.
	filesByID map[robustio.FileID][]*diskFile
}

func newMemoizedFS() *memoizedFS {
	return &memoizedFS{filesByID: make(map[robustio.FileID][]*diskFile)}
}

// A diskFile is a file in the filesystem, or a failure to read one.
// It implements the file.Source interface.
//
// A diskFile does not hold the file's content: snapshots keep a handle for
// every file in the workspace, and most of them are only read again on a cache
// miss, so holding every file's bytes would cost as much memory as the source
// tree. Content rereads the file and checks it against the hash taken when the
// handle was created.
type diskFile struct {
	uri     protocol.DocumentURI
	modTime time.Time
	size    int
	hash    file.Hash
	err     error
}

func (h *diskFile) String() string { return h.uri.Path() }

func (h *diskFile) URI() protocol.DocumentURI { return h.uri }

func (h *diskFile) Identity() file.Identity {
	return file.Identity{
		URI:  h.uri,
		Hash: h.hash,
	}
}

func (h *diskFile) SameContentsOnDisk() bool    { return true }
func (h *diskFile) Version() int32              { return 0 }
func (h *diskFile) ModTime() (time.Time, error) { return h.modTime, h.err }

// Content rereads the file. It reports an error, rather than returning
// content that does not match the handle's hash, if the file changed after the
// handle was created: results derived from content are cached by hash, and the
// client reports the change, which replaces the handle.
func (h *diskFile) Content() ([]byte, error) {
	if h.err != nil {
		return nil, h.err
	}
	content, err := readContent(h.uri)
	if err != nil {
		return nil, err
	}
	if file.HashOf(content) != h.hash {
		return nil, fmt.Errorf("%s changed on disk after it was read", h.uri.Path())
	}
	return content, nil
}

// ReadFile stats and (maybe) reads the file, updates the cache, and returns it.
func (fs *memoizedFS) ReadFile(ctx context.Context, uri protocol.DocumentURI) (file.Handle, error) {
	id, mtime, err := robustio.GetFileID(uri.Path())
	if err != nil {
		// file does not exist
		return &diskFile{
			err: err,
			uri: uri,
		}, nil
	}

	// We check if the file has changed by comparing modification times. Notably,
	// this is an imperfect heuristic as various systems have low resolution
	// mtimes (as much as 1s on WSL or s390x builders), so we only cache
	// filehandles if mtime is old enough to be reliable, meaning that we don't
	// expect a subsequent write to have the same mtime.
	//
	// The coarsest mtime precision we've seen in practice is 1s, so consider
	// mtime to be unreliable if it is less than 2s old. Capture this before
	// doing anything else.
	recentlyModified := time.Since(mtime) < 2*time.Second

	fs.mu.Lock()
	fhs, ok := fs.filesByID[id]
	if ok && fhs[0].modTime.Equal(mtime) {
		var fh *diskFile
		// We have already seen this file and it has not changed.
		for _, h := range fhs {
			if h.uri == uri {
				fh = h
				break
			}
		}
		// No file handle for this exact URI. Create an alias, but share content.
		if fh == nil {
			newFH := *fhs[0]
			newFH.uri = uri
			fh = &newFH
			fhs = append(fhs, fh)
			fs.filesByID[id] = fhs
		}
		fs.mu.Unlock()
		return fh, nil
	}
	fs.mu.Unlock()

	// Unknown file, or file has changed. Read (or re-read) it.
	fh, err := readFile(ctx, uri, mtime) // ~25us
	if err != nil {
		return nil, err // e.g. cancelled (not: read failed)
	}

	fs.mu.Lock()
	if !recentlyModified {
		fs.filesByID[id] = []*diskFile{fh}
	} else {
		delete(fs.filesByID, id)
	}
	fs.mu.Unlock()
	return fh, nil
}

// fileStats returns information about the set of files stored in fs. It is
// intended for debugging only.
func (fs *memoizedFS) fileStats() (files, largest, errs int) {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	files = len(fs.filesByID)
	largest = 0
	errs = 0

	for _, files := range fs.filesByID {
		rep := files[0]
		if rep.size > largest {
			largest = rep.size
		}
		if rep.err != nil {
			errs++
		}
	}
	return files, largest, errs
}

// ioLimit limits the number of parallel file reads per process.
var ioLimit = make(chan struct{}, 128)

func readFile(ctx context.Context, uri protocol.DocumentURI, mtime time.Time) (*diskFile, error) {
	select {
	case ioLimit <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	defer func() { <-ioLimit }()

	ctx, done := event.Start(ctx, "cache.readFile", label.File.Of(uri.Path()))
	_ = ctx
	defer done()

	// It is possible that a race causes us to read a file with different file
	// ID, or whose mtime differs from the given mtime. However, in these cases
	// we expect the client to notify of a subsequent file change, and the file
	// content should be eventually consistent.
	content, err := os.ReadFile(uri.Path()) // ~20us
	if err != nil {
		content = nil // just in case
	}
	return &diskFile{
		modTime: mtime,
		uri:     uri,
		size:    len(content),
		hash:    file.HashOf(content),
		err:     err,
	}, nil
}

// readContent rereads a file for diskFile.Content, sharing the limit on
// parallel reads.
func readContent(uri protocol.DocumentURI) ([]byte, error) {
	ioLimit <- struct{}{}
	defer func() { <-ioLimit }()
	return os.ReadFile(uri.Path())
}
