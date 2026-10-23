#!/usr/bin/env python3
"""Header/layout validation and actual reflink/payload tests; no model needed."""
import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('joiner', Path(__file__).resolve().parents[1] / 'gguf-tools/deepseek41_join.py')
j = importlib.util.module_from_spec(spec)
spec.loader.exec_module(j)


def kv(key, value):
    if isinstance(value, str):
        return j.wstr(key) + struct.pack('<I', 8) + j.wstr(value)
    return j.wstr(key) + struct.pack('<IQ', 10, value)


def shard(path, number, tensors, count=3, total=3, metadata=None):
    entries = {'split.no': number, 'split.count': count, 'split.tensors.count': total}
    if number == 0:
        entries.update({'general.architecture': 'deepseek41', 'general.alignment': 32})
    entries.update(metadata or {})
    data = bytearray()
    directory = bytearray()
    for name, kind, dims, payload in tensors:
        offset = j.align(len(data), 32)
        data.extend(bytes(offset - len(data)))
        directory.extend(j.wstr(name) + struct.pack('<I', len(dims)) +
                         struct.pack('<' + 'Q' * len(dims), *dims) + struct.pack('<IQ', kind, offset))
        data.extend(payload)
    header = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(entries)) + b''.join(kv(k, v) for k, v in entries.items()) + directory
    path.write_bytes(header + bytes(j.align(len(header), 32) - len(header)) + data)
    return path


class JoinTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.paths = [self.root / f'shard-{i}.gguf' for i in range(3)]
        shard(self.paths[0], 0, [('blk.0.ffn_gate_exps.weight', 39, [32, 2, 1], bytes(range(34)))])
        for i, layer in ((1, 1), (2, 14)):
            shard(self.paths[i], i, [(f'blk.{layer}.engram_embd.weight', 24, [264, 32], bytes([i]) * (264*32))])

    def tearDown(self):
        self.tmp.cleanup()

    def layout(self):
        fps = [p.open('rb') for p in self.paths]
        try:
            return j.layout(fps, 4096)
        finally:
            for fp in fps:
                fp.close()

    def test_offsets_and_metadata(self):
        headers, raw, size = self.layout()
        self.assertEqual(len(headers), 3)
        self.assertNotIn(b'split.', raw)
        for h in headers:
            self.assertEqual(h['base'] % 4096, 0)
        self.assertGreater(size, sum(p.stat().st_size for p in self.paths))

    def test_payload_reflink(self):
        output = self.root / 'joined.gguf'
        receipt = j.join(self.paths, output)
        self.assertEqual(receipt['tensors'], 3)
        with output.open('rb') as fp:
            merged = j.read_header(fp)
            for source in self.paths:
                with source.open('rb') as sf:
                    original = j.read_header(sf)
                    for before in original['tensors']:
                        after = next(t for t in merged['tensors'] if t['name'] == before['name'])
                        self.assertEqual(before['shape'], after['shape'])
                        self.assertEqual(before['type'], after['type'])
                        sf.seek(original['data'] + before['offset'])
                        fp.seek(merged['data'] + after['offset'])
                        self.assertEqual(sf.read(before['bytes']), fp.read(after['bytes']))
        with self.assertRaisesRegex(ValueError, 'exists'):
            j.join(self.paths, output)

    def test_missing_or_wrong_order(self):
        self.paths.reverse()
        with self.assertRaisesRegex(ValueError, 'out-of-order'):
            self.layout()

    def test_duplicate_name(self):
        shard(self.paths[1], 1, [('blk.0.ffn_gate_exps.weight', 39, [32, 2, 1], bytes(34))])
        with self.assertRaisesRegex(ValueError, 'Duplicate tensor across'):
            self.layout()

    def test_truncated_payload(self):
        os.truncate(self.paths[0], self.paths[0].stat().st_size - 1)
        with self.assertRaisesRegex(ValueError, 'truncated tensor'):
            self.layout()

    def test_bad_split_total(self):
        shard(self.paths[1], 1, [('blk.1.engram_embd.weight', 24, [264, 1], bytes(264))], total=4)
        with self.assertRaisesRegex(ValueError, 'count mismatch'):
            self.layout()

    def test_unknown_type(self):
        shard(self.paths[0], 0, [('bad', 9999, [32], bytes(32))])
        with self.assertRaisesRegex(ValueError, 'Unsupported tensor type'):
            self.layout()

    def test_resident_after_engram(self):
        shard(self.paths[2], 2, [('resident', 0, [32], bytes(128))])
        with self.assertRaisesRegex(ValueError, 'both trailing Engram'):
            self.layout()

    def test_second_shard_metadata_rejected(self):
        shard(self.paths[1], 1, [('blk.1.engram_embd.weight', 24, [264, 1], bytes(264))], metadata={'general.architecture': 'other'})
        with self.assertRaisesRegex(ValueError, 'Unexpected metadata'):
            self.layout()

    def test_existing_incomplete_not_overwritten(self):
        output = self.root / 'joined.gguf'
        partial = output.with_name(output.name + '.incomplete')
        partial.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):
            j.join(self.paths, output)
        self.assertEqual(partial.read_bytes(), b'keep')


if __name__ == '__main__':
    unittest.main(verbosity=2)
