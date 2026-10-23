#!/usr/bin/env python3
"""Join split DeepSeek V4.1 GGUFs with Linux reflinks; never copy the payload."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys

FICLONERANGE = 0x4020940D
MAX_HEADER = 64 << 20
BLOCKS = {0: (1, 4), 1: (1, 2), 8: (32, 34), 24: (1, 1), 39: (32, 17)}
SCALARS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
           6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}


def align(n, a):
    return (n + a - 1) // a * a


def wstr(s):
    b = s.encode('utf-8')
    return struct.pack('<Q', len(b)) + b


def identity(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


class Reader:
    def __init__(self, fp):
        self.fp, self.pos = fp, 0

    def read(self, n):
        if n < 0 or self.pos + n > MAX_HEADER:
            raise ValueError('GGUF header exceeds 64 MiB')
        b = self.fp.read(n)
        if len(b) != n:
            raise ValueError('Truncated GGUF header')
        self.pos += n
        return b

    def num(self, fmt):
        return struct.unpack('<' + fmt, self.read(struct.calcsize('<' + fmt)))[0]

    def string(self):
        return self.read(self.num('Q')).decode('utf-8')

    def value(self, kind, keep=True):
        if kind in SCALARS:
            return self.num(SCALARS[kind])
        if kind == 8:
            return self.string()
        if kind == 9:
            element, count = self.num('I'), self.num('Q')
            if element == 9 or count > MAX_HEADER:
                raise ValueError('Invalid GGUF array')
            for _ in range(count):
                self.value(element, False)
            return None
        raise ValueError(f'Unknown GGUF metadata type {kind}')


def read_header(fp):
    fp.seek(0)
    r = Reader(fp)
    if r.read(4) != b'GGUF' or r.num('I') != 3:
        raise ValueError('Expected GGUF v3')
    nt, nkv = r.num('Q'), r.num('Q')
    if not 0 < nt <= 100000 or nkv > 100000:
        raise ValueError('Invalid GGUF counts')
    kv, rawkv = {}, []
    for _ in range(nkv):
        begin = r.pos
        key = r.string()
        if key in kv:
            raise ValueError(f'Duplicate metadata {key}')
        kind = r.num('I')
        kv[key] = r.value(kind)
        end = r.pos
        fp.seek(begin)
        raw = fp.read(end - begin)
        fp.seek(end)
        rawkv.append((key, raw))
    alignment = kv.get('general.alignment', 32)
    if not isinstance(alignment, int) or alignment < 1 or alignment > 4096 or alignment & (alignment - 1):
        raise ValueError('Unsupported GGUF alignment')
    tensors, names = [], set()
    for _ in range(nt):
        name, ndim = r.string(), r.num('I')
        if name in names or not 1 <= ndim <= 4:
            raise ValueError('Duplicate tensor or invalid rank')
        names.add(name)
        dims = [r.num('Q') for _ in range(ndim)]
        kind, offset = r.num('I'), r.num('Q')
        if kind not in BLOCKS:
            raise ValueError(f'Unsupported tensor type {kind} ({name})')
        block, width = BLOCKS[kind]
        if any(not d for d in dims) or dims[0] % block or offset % alignment:
            raise ValueError(f'Invalid tensor geometry/alignment: {name}')
        size = math.prod(dims) // block * width
        tensors.append(dict(name=name, shape=dims, type=kind, offset=offset, bytes=size))
    data = align(r.pos, alignment)
    size = os.fstat(fp.fileno()).st_size
    end = 0
    for t in sorted(tensors, key=lambda t: t['offset']):
        if t['offset'] < end or data + t['offset'] + t['bytes'] > size:
            raise ValueError(f'Overlapping/truncated tensor: {t["name"]}')
        end = t['offset'] + t['bytes']
    return dict(kv=kv, rawkv=rawkv, tensors=tensors, data=data, size=size, alignment=alignment)


def layout(files, page):
    if not files or page < 4096 or page & (page - 1):
        raise ValueError('Expected input shards and a power-of-two page size')
    headers = []
    for index, fp in enumerate(files):
        st = os.fstat(fp.fileno())
        if not stat.S_ISREG(st.st_mode):
            raise ValueError('Input is not a regular file')
        h = read_header(fp)
        h['identity'] = identity(st)
        kv = h['kv']
        if kv.get('split.no') != index or kv.get('split.count') != len(files):
            raise ValueError('Incomplete or out-of-order shard set')
        if index and set(kv) != {'split.no', 'split.count', 'split.tensors.count'}:
            raise ValueError('Unexpected metadata outside first shard')
        headers.append(h)
    first = headers[0]
    if first['kv'].get('general.architecture') != 'deepseek41':
        raise ValueError('Expected deepseek41 architecture')
    count = sum(len(h['tensors']) for h in headers)
    if any(h['kv'].get('split.tensors.count') != count for h in headers):
        raise ValueError('Split tensor count mismatch')
    names = [t['name'] for h in headers for t in h['tensors']]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate tensor across shards')
    rawkv = [v for k, v in first['rawkv'] if not k.startswith('split.')]
    prefix = b'GGUF' + struct.pack('<IQQ', 3, count, len(rawkv)) + b''.join(rawkv)
    directory_size = sum(8 + len(t['name'].encode()) + 4 + 8 * len(t['shape']) + 4 + 8
                         for h in headers for t in h['tensors'])
    data = align(len(prefix) + directory_size, first['alignment'])
    base = align(data, page)
    directory, extents = bytearray(), []
    for h in headers:
        h['base'] = base
        for t in h['tensors']:
            absolute = base + h['data'] + t['offset']
            relative = absolute - data
            if relative % first['alignment']:
                raise ValueError('Shard alignment incompatible with first shard')
            directory.extend(wstr(t['name']) + struct.pack('<I', len(t['shape'])) +
                             struct.pack('<' + 'Q' * len(t['shape']), *t['shape']) +
                             struct.pack('<IQ', t['type'], relative))
            extents.append((absolute, absolute + t['bytes'], t['name']))
        base = align(base + h['size'], page)
    engram = {name: (a, b) for a, b, name in extents if name in
              ('blk.1.engram_embd.weight', 'blk.14.engram_embd.weight')}
    if len(engram) != 2:
        raise ValueError('Expected both trailing Engram tables')
    a, b = engram['blk.1.engram_embd.weight']
    c, _ = engram['blk.14.engram_embd.weight']
    resident_end = max(end for _, end, name in extents if name not in engram)
    if resident_end > a // page * page or c < b:
        raise ValueError('Engram cannot be safely unmapped after resident weights')
    header = prefix + directory
    return headers, header, base


def join(paths, output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('Output already exists')
    temporary = output.with_name(output.name + '.incomplete')
    files = []
    try:
        for path in paths:
            files.append(open(path, 'rb'))
        page = os.sysconf('SC_PAGESIZE')
        headers, header, size = layout(files, page)
        # Never fall back to copy_file_range/copy. Model bytes must stay shared.
        with temporary.open('xb') as target:
            target.write(header)
            target.flush()
            os.ftruncate(target.fileno(), size)
            for path, source, h in zip(paths, files, headers):
                if identity(os.fstat(source.fileno())) != h['identity']:
                    raise ValueError(f'Input changed: {path}')
                length = h['size'] // page * page
                if length:
                    try:
                        fcntl.ioctl(target.fileno(), FICLONERANGE,
                                    struct.pack('=qQQQ', source.fileno(), 0, length, h['base']))
                    except OSError as exc:
                        raise RuntimeError('Reflink required on the same supported filesystem; '
                                           'no payload-copy fallback was attempted') from exc
                tail = os.pread(source.fileno(), h['size'] - length, length)
                if len(tail) != h['size'] - length:
                    raise ValueError('Source tail truncated')
                if tail and os.pwrite(target.fileno(), tail, h['base'] + length) != len(tail):
                    raise OSError('Short tail write')
                if identity(os.fstat(source.fileno())) != h['identity']:
                    raise ValueError(f'Input changed: {path}')
                print(f'Reflinked {Path(path).name}: {h["size"]:,} bytes', flush=True)
            os.fsync(target.fileno())
        with temporary.open('rb') as fp:
            merged = read_header(fp)
        if len(merged['tensors']) != sum(len(h['tensors']) for h in headers):
            raise ValueError('Joined tensor count mismatch')
        # link() refuses replacement atomically; never overwrite a raced output.
        os.link(temporary, output)
        temporary.unlink()
        return dict(output=str(output.resolve()), bytes=size, shards=len(files),
                    tensors=len(merged['tensors']), payload='unchanged, reflinked',
                    integrity='input header/ranges/stat + ioctl; no full payload scan')
    finally:
        for fp in files:
            fp.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('shards', nargs='+', type=Path, help='All shards in split.no order')
    args = parser.parse_args()
    try:
        print(json.dumps(join(args.shards, args.output), indent=2))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f'{exc}\nTask-owned .incomplete output, if present, is retained for inspection.\n')


if __name__ == '__main__':
    main()
