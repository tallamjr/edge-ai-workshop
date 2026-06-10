#!/usr/bin/env python3
"""
scripts/common/split_model.py

Analyze a neutron-compiled TFLite model and optionally split it into
separate sub-models for pipelined CPU / NPU execution.

After neutron-converter, some ops may remain on CPU (quantize, dequantize,
reshape ops at boundaries, or ops not yet supported by Neutron). Splitting
these segments and pipelining their execution overlaps CPU and NPU work,
improving throughput.

The split is described by a manifest JSON file. The board-side inference
code reads the manifest at startup: a single .tflite path behaves exactly
as before; a .json path activates multi-stage pipelined execution with no
additional configuration.

Usage — analysis only:
    python scripts/common/split_model.py --model models/deploy/yolov8n_neutron.tflite

Usage — split:
    python scripts/common/split_model.py \\
        --model  models/deploy/yolov8n_neutron.tflite \\
        --split  \\
        --output-dir models/work/split/

Outputs (when --split):
    models/work/split/pre.tflite      CPU ops before NeutronGraph (omitted if none)
    models/work/split/npu.tflite      NeutronGraph only
    models/work/split/post.tflite     CPU ops after NeutronGraph (omitted if none)
    models/work/split/pipeline.json   Manifest — point model.path here on the board

Requires tflite-extractor (part of the NXP eIQ Toolkit) on PATH for --split.
"""

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path


# ============================================================
# Minimal FlatBuffer reader (no generated schema bindings)
# ============================================================

def _ru32(b, o): return struct.unpack_from('<I', b, o)[0]
def _ri32(b, o): return struct.unpack_from('<i', b, o)[0]
def _ru16(b, o): return struct.unpack_from('<H', b, o)[0]


class _T:
    """Read-only FlatBuffer table."""
    __slots__ = ('_b', '_p', '_vt', '_vs')

    def __init__(self, buf, pos):
        self._b  = buf
        self._p  = pos
        vd       = _ri32(buf, pos)
        self._vt = pos - vd
        self._vs = _ru16(buf, self._vt)

    def _fp(self, vto):
        if vto >= self._vs: return None
        fo = _ru16(self._b, self._vt + vto)
        return (self._p + fo) if fo else None

    def u8  (self, v, d=0):      p = self._fp(v); return self._b[p]                                if p else d
    def i32 (self, v, d=0):      p = self._fp(v); return _ri32(self._b, p)                         if p else d
    def u32 (self, v, d=0):      p = self._fp(v); return _ru32(self._b, p)                         if p else d
    def bool_(self, v, d=False): p = self._fp(v); return bool(self._b[p])                          if p else d

    def str_(self, v):
        p = self._fp(v)
        if p is None: return None
        s = p + _ru32(self._b, p)
        n = _ru32(self._b, s)
        return self._b[s+4:s+4+n].decode('utf-8', errors='replace')

    def tbl(self, v):
        p = self._fp(v)
        if p is None: return None
        return _T(self._b, p + _ru32(self._b, p))

    def tvec(self, v):
        p = self._fp(v)
        if p is None: return []
        vec = p + _ru32(self._b, p)
        n   = _ru32(self._b, vec)
        out = []
        for i in range(n):
            ip = vec + 4 + i*4
            out.append(_T(self._b, ip + _ru32(self._b, ip)))
        return out

    def ivec(self, v):
        p = self._fp(v)
        if p is None: return []
        vec = p + _ru32(self._b, p)
        n   = _ru32(self._b, vec)
        return list(struct.unpack_from(f'<{n}i', self._b, vec+4))

    def fvec(self, v):
        p = self._fp(v)
        if p is None: return []
        vec = p + _ru32(self._b, p)
        n   = _ru32(self._b, vec)
        return list(struct.unpack_from(f'<{n}f', self._b, vec+4))

    def i64vec(self, v):
        p = self._fp(v)
        if p is None: return []
        vec = p + _ru32(self._b, p)
        n   = _ru32(self._b, vec)
        return list(struct.unpack_from(f'<{n}q', self._b, vec+4))

    def bvec(self, v):
        p = self._fp(v)
        if p is None: return b''
        vec = p + _ru32(self._b, p)
        n   = _ru32(self._b, vec)
        return bytes(self._b[vec+4:vec+4+n])


# TFLite FlatBuffer VTable offsets (from tensorflow/lite/schema/schema.fbs)
_M_VERSION   = 4;  _M_OPCODES  = 6;  _M_SUBGRAPHS = 8;  _M_BUFFERS   = 12
_SG_TENSORS  = 4;  _SG_INPUTS  = 6;  _SG_OUTPUTS  = 8;  _SG_OPS      = 10
_T_SHAPE     = 4;  _T_TYPE     = 6;  _T_BUFFER    = 8;  _T_NAME      = 10
_T_QUANT     = 12; _T_ISVAR    = 14
_Q_SCALE     = 8;  _Q_ZP       = 10; _Q_QDIM      = 16
_OP_OPCODE   = 4;  _OP_IN      = 6;  _OP_OUT      = 8
_OC_DEPR     = 4;  _OC_CUSTOM  = 6;  _OC_VER      = 8;  _OC_BUILTIN  = 10
_BUF_DATA    = 4


# ============================================================
# TFLite model parser
# ============================================================

def _parse_model(model_bytes: bytes) -> dict:
    buf = bytearray(model_bytes)
    m   = _T(buf, _ru32(buf, 0))

    opcodes = []
    for oc in m.tvec(_M_OPCODES):
        opcodes.append({'deprecated_builtin_code': oc.u8(_OC_DEPR, 0),
                        'custom_code': oc.str_(_OC_CUSTOM),
                        'version':     oc.i32(_OC_VER, 1),
                        'builtin_code': oc.i32(_OC_BUILTIN, 0)})

    raw_bufs = [bf.bvec(_BUF_DATA) for bf in m.tvec(_M_BUFFERS)]

    sg = m.tvec(_M_SUBGRAPHS)[0]
    model_inputs  = sg.ivec(_SG_INPUTS)
    model_outputs = sg.ivec(_SG_OUTPUTS)

    tensors = []
    for t in sg.tvec(_SG_TENSORS):
        buf_i = t.u32(_T_BUFFER, 0)
        q     = t.tbl(_T_QUANT)
        tensors.append({
            'shape':       t.ivec(_T_SHAPE),
            'dtype':       t.u8(_T_TYPE, 0),
            'buffer':      buf_i,
            'name':        t.str_(_T_NAME) or '',
            'scales':      q.fvec(_Q_SCALE)    if q else [],
            'zero_points': q.i64vec(_Q_ZP)     if q else [],
            'quant_dim':   q.i32(_Q_QDIM, 0)   if q else 0,
            'is_variable': t.bool_(_T_ISVAR, False),
            'data':        raw_bufs[buf_i] if buf_i and buf_i < len(raw_bufs) else b'',
        })

    operators = []
    for op in sg.tvec(_SG_OPS):
        operators.append({
            'opcode_index': op.u32(_OP_OPCODE, 0),
            'inputs':       op.ivec(_OP_IN),
            'outputs':      op.ivec(_OP_OUT),
        })

    return {'version': m.u32(_M_VERSION, 3),
            'operator_codes': opcodes, 'tensors': tensors,
            'operators': operators, 'model_inputs': model_inputs,
            'model_outputs': model_outputs}


# ============================================================
# Analysis  (FlatBuffer-based — no TFLite interpreter needed,
#            so it works on the host without libneutron_delegate.so)
# ============================================================

# Minimal map of TFLite builtin op codes to readable names.
# Full list: tensorflow/lite/schema/schema_generated.h  enum BuiltinOperator
_BUILTIN_OP_NAMES: dict[int, str] = {
    0: 'ADD', 1: 'AVERAGE_POOL_2D', 2: 'CONCATENATION', 3: 'CONV_2D',
    4: 'DEPTHWISE_CONV_2D', 6: 'DEQUANTIZE', 9: 'FULLY_CONNECTED',
    14: 'LOGISTIC', 16: 'MAX_POOL_2D', 18: 'MUL', 19: 'RELU',
    21: 'RESHAPE', 22: 'RESIZE_BILINEAR', 24: 'SOFTMAX',
    25: 'SPACE_TO_DEPTH', 27: 'TANH', 37: 'TRANSPOSE',
    39: 'SUB', 40: 'DIV', 41: 'SQUEEZE', 44: 'PAD',
    50: 'TRANSPOSE_CONV', 53: 'POW', 55: 'ARG_MAX',
    82: 'MIRROR_PAD', 86: 'HARD_SWISH', 96: 'SPLIT_V',
    114: 'QUANTIZE',
}


def analyze(model_path: str) -> dict:
    """
    Return op-distribution info for a neutron-compiled TFLite model.

    Uses direct FlatBuffer parsing — no TFLite interpreter or NPU delegate
    required, so this runs correctly on the host machine.

    Keys: total_ops, neutron_indices, pre_cpu, post_cpu, can_split, ops,
          npu_in_tensors, npu_out_tensors, model_inputs, model_outputs.
    """
    parsed   = _parse_model(Path(model_path).read_bytes())
    opcodes  = parsed['operator_codes']
    operators = parsed['operators']

    ops     = []
    neutron = []
    for i, op in enumerate(operators):
        oc   = opcodes[op['opcode_index']]
        cc   = oc['custom_code'] or ''
        if cc and 'Neutron' in cc:
            name = cc
            neutron.append(i)
        elif cc:
            name = f'CUSTOM({cc})'
        else:
            name = _BUILTIN_OP_NAMES.get(oc['builtin_code'],
                                         f'BUILTIN({oc["builtin_code"]})')
        ops.append({'index': i, 'name': name,
                    'inputs': op['inputs'], 'outputs': op['outputs']})

    first_npu = min(neutron, default=0)
    last_npu  = max(neutron, default=-1)
    pre  = [o for o in ops if o['index'] < first_npu]
    post = [o for o in ops if o['index'] > last_npu]

    npu_in_tensors  = []
    npu_out_tensors = []
    if neutron:
        npu_op = parsed['operators'][neutron[0]]
        npu_in_tensors  = [ti for ti in npu_op['inputs']  if ti >= 0]
        npu_out_tensors = [ti for ti in npu_op['outputs'] if ti >= 0]

    return {'total_ops': len(ops), 'neutron_indices': neutron,
            'pre_cpu': pre, 'post_cpu': post,
            'can_split': len(neutron) == 1 and (pre or post),
            'ops': ops,
            'npu_in_tensors':  npu_in_tensors,
            'npu_out_tensors': npu_out_tensors,
            'model_inputs':    parsed['model_inputs'],
            'model_outputs':   parsed['model_outputs']}


def print_analysis(r: dict) -> None:
    print()
    print("  Op distribution")
    print("  " + "─" * 44)
    for op in r['ops']:
        tag = "NPU" if op['index'] in r['neutron_indices'] else "CPU"
        print(f"  [{op['index']:2d}] {tag}  {op['name']}")
    print()
    print(f"  Total ops    : {r['total_ops']}")
    print(f"  NeutronGraph : {len(r['neutron_indices'])}")
    print(f"  Pre-CPU ops  : {len(r['pre_cpu'])}")
    print(f"  Post-CPU ops : {len(r['post_cpu'])}")
    if r['npu_in_tensors']:
        print(f"  NPU inputs   : {r['npu_in_tensors']}")
        print(f"  NPU outputs  : {r['npu_out_tensors']}")
    print()
    if r['can_split']:
        print("  [OK] Good candidate for pipelined split. Run with --split.")
    elif len(r['neutron_indices']) > 1:
        print("  [INFO] Multiple NeutronGraph ops — manual split strategy needed.")
    elif not r['neutron_indices']:
        print("  [WARN] No NeutronGraph op found. Was the delegate loaded during conversion?")
    else:
        print("  [INFO] No pre/post CPU ops — model already fully on NPU. No split needed.")
    print()


# ============================================================
# Split — delegates to tflite-extractor (NXP eIQ Toolkit)
# ============================================================

def _check_extractor() -> str:
    """Return the tflite-extractor path or raise if not found."""
    import shutil
    path = shutil.which('tflite-extractor')
    if path:
        return path
    raise RuntimeError(
        "tflite-extractor not found on PATH.\n"
        "Install the NXP eIQ Toolkit and add it to PATH:\n"
        "  export PATH=$PATH:/path/to/eiq-neutron-sdk/bin")


def _extract(extractor: str, model_path: str, out_path: Path,
             in_tensors: list[int], out_tensors: list[int]) -> None:
    """Run tflite-extractor to cut a sub-graph from model_path."""
    cmd = [
        extractor,
        '--input',  model_path,
        '--output', str(out_path),
        f'--input_tensors={",".join(map(str, in_tensors))}',
        f'--output_tensors={",".join(map(str, out_tensors))}',
    ]
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"tflite-extractor failed (exit {result.returncode}):\n"
            f"{result.stderr.strip()}")
    print(f"  [OK] {out_path}  ({out_path.stat().st_size // 1024} KB)")


def split(model_path: str, output_dir: str, analysis: dict | None = None) -> Path:
    """
    Split the model into sub-models using tflite-extractor and write a
    pipeline manifest JSON.  Returns the path to the manifest.

    Pass a pre-computed analysis dict to avoid re-parsing the model.
    Requires tflite-extractor (NXP eIQ Toolkit) on PATH.
    """
    extractor = _check_extractor()

    r = analysis if analysis is not None else analyze(model_path)
    if not r['can_split']:
        raise ValueError(
            "Cannot auto-split: need exactly one NeutronGraph and at "
            "least one pre or post CPU op.")

    out      = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    m_in         = r['model_inputs']
    m_out        = r['model_outputs']
    npu_in       = r['npu_in_tensors']
    npu_out      = r['npu_out_tensors']
    pre_ops      = r['pre_cpu']
    post_ops     = r['post_cpu']
    pipeline     = []

    if pre_ops:
        _extract(extractor, model_path, out / "pre.tflite", m_in, npu_in)
        pipeline.append({"label": "pre", "file": "pre.tflite", "use_npu": False})

    _extract(extractor, model_path, out / "npu.tflite",
             npu_in  if pre_ops  else m_in,
             npu_out if post_ops else m_out)
    pipeline.append({"label": "npu", "file": "npu.tflite", "use_npu": True})

    if post_ops:
        _extract(extractor, model_path, out / "post.tflite", npu_out, m_out)
        pipeline.append({"label": "post", "file": "post.tflite", "use_npu": False})

    manifest_path = out / "pipeline.json"
    manifest_path.write_text(json.dumps({"pipeline": pipeline}, indent=2))
    print(f"  [OK] {manifest_path}")
    return manifest_path


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Analyze / split a neutron-compiled TFLite for CPU/NPU pipelining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model",      required=True, help="Neutron-compiled .tflite path")
    ap.add_argument("--split",      action="store_true", help="Write split sub-models")
    ap.add_argument("--output-dir", default=None,
                    help="Output dir for split files (default: <model_dir>/split/)")
    args = ap.parse_args()

    if not Path(args.model).exists():
        print(f"[ERROR] Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Analyzing: {args.model}")
    r = analyze(args.model)
    print_analysis(r)

    if not args.split:
        return

    if not r['can_split']:
        print("[INFO] Nothing to split.")
        return

    try:
        _check_extractor()
    except RuntimeError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.output_dir or str(Path(args.model).parent / "split")
    print(f"  Splitting into: {out_dir}/\n")
    manifest = split(args.model, out_dir, analysis=r)
    print(f"\n  Point model.path in config.json to:\n    {manifest}")
    print("  Then deploy with: make model-split-deploy\n")


if __name__ == "__main__":
    main()
