"""VLLM-01 启动日志解析：把一份 vllm serve 的启动日志压成一张对账表。

    python3 parse_startup_log.py logs/serve-20261029-101500.log
    python3 parse_startup_log.py logs/serve-*.log --model qwen3-8b --gpu h100-sxm   # 附带和账本对账
    python3 parse_startup_log.py sample_startup.log --json

抽取的行全部取自 vLLM 主干 0d6f92a 的 logger 格式串（文件路径见每条正则旁的注释），
版本升级后如果某一项变成“未找到”，先去对应文件确认格式串有没有改。
日志前缀格式为 (进程名 pid=N) LEVEL MM-DD HH:MM:SS [文件:行号] 消息，见 vllm/logger.py 的 _FORMAT；
INFO 级别下“文件”只是文件名，VLLM_LOGGING_LEVEL=DEBUG 时才换成缩写后的相对路径。
多行消息（ASCII logo、长报错）每行都会补上同样的前缀，行尾带 \r，所以读行时要连 \r 一起去掉。
"""
import argparse
import json
import os
import re
import sys

PREFIX = re.compile(r'^(?:\S*)?\((?P<proc>[\w:]+) pid=(?P<pid>\d+)\) (?P<level>[A-Z]+) '
                    r'(?P<ts>\d\d-\d\d \d\d:\d\d:\d\d) \[(?P<src>[^\]]+)\] (?P<msg>.*)$')

# (键, 正则, 说明)。正则只匹配 msg 部分；多次出现时保留第一次（TP>1 时每个 rank 都会打一遍）。
RULES = [
    ('version', r'vLLM server version (\S+), serving model (\S+)', 'entrypoints/common/startup.py'),
    ('version_logo', r'█\s+version\s+(\S+)$', 'entrypoints/common/startup.py（ASCII logo 模式）'),
    ('model_logo', r'█\s+model\s+(\S+)$', 'entrypoints/common/startup.py（ASCII logo 模式）'),
    ('non_default', r'non-default args: (.*)$', 'entrypoints/common/startup.py'),
    ('arch', r'Resolved architecture: (\S+)', 'config/model.py'),
    ('max_model_len', r'Using max model len (\d+)', 'config/model.py'),
    ('max_batched_tokens', r'Chunked prefill is enabled with max_num_batched_tokens=(\d+)', 'config/scheduler.py'),
    ('attn_backend', r'Using (\S+) attention backend out of potential backends: (.*)\.$', 'platforms/cuda.py'),
    ('mrv2', r'(Using V2 Model Runner|Model Runner V2 .*using the V1 model runner.*)', 'v1/worker/gpu_worker.py, config/vllm.py'),
    ('compile_cache_dir', r'Using cache directory: (\S+) for vLLM\'s torch\.compile', 'compilation/backends.py'),
    ('compile_miss', r'Compiling a graph for compile range (.*) takes ([\d.]+) s', 'compilation/backends.py'),
    ('load_weights_s', r'Loading weights took ([\d.]+) seconds', 'model_executor/model_loader/default_loader.py'),
    ('model_load', r'Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds', 'v1/worker/gpu_model_runner.py'),
    ('dynamo_s', r'Dynamo bytecode transform time: ([\d.]+) s', 'compilation/backends.py'),
    ('compile_cache_hit', r'Directly load the compiled graph\(s\) for compile range (.*) from the cache, took ([\d.]+) s',
     'compilation/backends.py'),
    ('compile_total_s', r'torch\.compile took ([\d.]+) s in total', 'compilation/monitor.py'),
    ('mem_profile', r'Memory profiling takes ([\d.]+) seconds\. Total non KV cache memory: ([\d.]+)GiB; '
                    r'torch peak memory increase: ([\d.]+)GiB; total consumed \(from mem_get_info\): ([\d.]+)GiB; '
                    r'weights memory: ([\d.]+)GiB', 'utils/mem_utils.py（DEBUG 级别）'),
    ('kv_avail_gib', r'Available KV cache memory: ([\d.]+) GiB', 'v1/worker/gpu_worker.py'),
    ('kv_tokens', r'(?:GPU|\w+)(?: \+ host)? KV cache size: ([\d,]+) tokens, '
                  r'Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x', 'v1/core/kv_cache_utils.py'),
    ('graph', r'Graph capturing finished in (\d+) secs, took ([\d.]+) GiB', 'v1/worker/gpu_model_runner.py'),
    ('graph_pool', r'CUDA graph pool memory: ([\d.]+) GiB \(actual\), ([\d.]+) GiB \(estimated\)', 'v1/worker/gpu_worker.py'),
    ('kv_hint', r'Replace gpu_memory_utilization config with `--kv-cache-memory=(\d+)`', 'v1/worker/gpu_worker.py'),
    ('init_engine', r'init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s(?: \(compilation: ([\d.]+) s)?',
     'v1/engine/core.py'),
    ('tasks', r'Supported tasks: (.*)$', 'entrypoints/launchers/api_server/entry.py'),
    ('listen', r'Starting vLLM server on (\S+)', 'entrypoints/launchers/api_server/entry.py'),
    # 失败路径
    ('err_no_mem', r'(No available memory for the cache blocks)', 'v1/core/kv_cache_planning.py'),
    ('err_max_len', r'To serve at least one request with the model\'s max seq len \((\d+)\), \(([\d.]+) GiB KV cache is needed, '
                    r'which is larger than the available KV cache memory \(([\d.]+) GiB\)', 'v1/core/kv_cache_planning.py'),
    ('err_free_mem', r'Free memory on device \(([\d.]+)/([\d.]+) GiB\) on startup is less than desired GPU memory utilization '
                     r'\(([\d.]+), ([\d.]+) GiB\)', 'v1/worker/utils.py'),
    ('err_ready_timeout', r'(Timed out waiting for engine core processes to start)', 'v1/engine/core_client.py'),
    ('warn_flashinfer', r'(FlashInfer\'s precompiled kernels are missing)', 'utils/flashinfer.py'),
]
COMPILED = [(k, re.compile(p), s) for k, p, s in RULES]


def parse(lines):
    found, procs, first_ts, last_ts = {}, {}, None, None
    for raw in lines:
        raw = raw.rstrip('\r\n')
        m = PREFIX.match(raw)
        msg = m.group('msg') if m else raw
        if m:
            procs.setdefault(m.group('proc'), m.group('pid'))
            first_ts = first_ts or m.group('ts')
            last_ts = m.group('ts')
        for k, rx, _ in COMPILED:
            if k in found:
                continue
            mm = rx.search(msg)
            if mm:
                found[k] = mm.groups()
    return found, procs, first_ts, last_ts


def _secs(ts):
    _, hms = ts.split(' ')
    h, mi, s = map(int, hms.split(':'))
    return h * 3600 + mi * 60 + s


def summarize(found, procs, first_ts, last_ts):
    def g(k, i=0, f=str):
        if k not in found or found[k][i] is None:
            return None
        v = found[k][i]
        return v if f is str else f(v.replace(',', ''))

    s = {
        'version': g('version') or g('version_logo'),
        'model': g('version', 1) or g('model_logo'),
        'arch': g('arch'),
        'processes': procs,
        'max_model_len': g('max_model_len', 0, int),
        'max_num_batched_tokens': g('max_batched_tokens', 0, int),
        'attention_backend': g('attn_backend'),
        'model_runner': ('V2' if g('mrv2') == 'Using V2 Model Runner' else ('V1（回退）' if 'mrv2' in found else None)),
        'weights_gib': g('model_load', 0, float),
        'model_load_s': g('model_load', 1, float),
        'torch_compile_s': g('compile_total_s', 0, float),
        'compile_cache_hit': 'compile_cache_hit' in found,
        'kv_avail_gib': g('kv_avail_gib', 0, float),
        'kv_tokens': g('kv_tokens', 0, int),
        'max_concurrency': g('kv_tokens', 2, float),
        'graph_capture_s': g('graph', 0, int),
        'graph_gib': g('graph', 1, float),
        'kv_cache_memory_hint': g('kv_hint', 0, int),
        'init_engine_s': g('init_engine', 0, float),
        'listen': g('listen'),
        'wall_s': (_secs(last_ts) - _secs(first_ts)) if first_ts and last_ts else None,
        'compile_cache_dir': g('compile_cache_dir'),
        'non_default_args': g('non_default'),
    }
    s['errors'] = [k for k in ('err_no_mem', 'err_max_len', 'err_free_mem', 'err_ready_timeout') if k in found]
    s['warnings'] = [k for k in ('warn_flashinfer',) if k in found]
    if 'err_max_len' in found:
        s['err_max_len'] = dict(max_model_len=g('err_max_len', 0, int), need_gib=g('err_max_len', 1, float),
                                avail_gib=g('err_max_len', 2, float))
    return s


ADVICE = {
    'err_no_mem': '权重加激活已经吃满 requested，先确认没有别的进程占卡，再调高 --gpu-memory-utilization 或换量化权重',
    'err_max_len': 'KV 放不下一条最长请求：调低 --max-model-len，或用 --max-model-len -1 让引擎自动收缩',
    'err_free_mem': '启动瞬间空闲显存低于 总显存 x util：卡上有别的进程，nvidia-smi 查掉它或调低 util',
    'err_ready_timeout': 'EngineCore 600 s 内没报就绪：多半是权重下载或加载太慢，调大 VLLM_ENGINE_READY_TIMEOUT_S 或预先下载',
    'warn_flashinfer': '没执行 vllm download-kernels，首启会现场编译 FlashInfer kernel，多花几分钟',
}


def show(s, ledger_cmp=None):
    rows = [
        ('版本 / 模型', f"{s['version']} / {s['model']}"),
        ('架构', s['arch']),
        ('进程', ', '.join(f'{k}({v})' for k, v in s['processes'].items())),
        ('max_model_len', s['max_model_len']),
        ('max_num_batched_tokens', s['max_num_batched_tokens']),
        ('注意力后端', s['attention_backend']),
        ('Model Runner', s['model_runner']),
        ('权重显存 GiB / 加载 s', f"{s['weights_gib']} / {s['model_load_s']}"),
        ('torch.compile s（缓存命中）', f"{s['torch_compile_s']}（{'是' if s['compile_cache_hit'] else '否'}）"
         if s['torch_compile_s'] is not None else None),
        ('KV 可用 GiB', s['kv_avail_gib']),
        ('KV 容量 token', f"{s['kv_tokens']:,}" if s['kv_tokens'] else None),
        ('max_model_len 下最大并发', s['max_concurrency']),
        ('CUDA Graph 捕获 s / GiB', f"{s['graph_capture_s']} / {s['graph_gib']}"),
        ('可回填 --kv-cache-memory', s['kv_cache_memory_hint']),
        ('init engine s', s['init_engine_s']),
        ('首行到末行 s', s['wall_s']),
        ('监听地址', s['listen']),
    ]
    w = max(len(k.encode('gbk', 'ignore')) for k, _ in rows) + 2
    for k, v in rows:
        pad = w - len(k.encode('gbk', 'ignore'))
        missing = v is None or str(v).replace('None', '').strip(' /') == ''
        print(f"  {k}{' ' * pad}{'未找到' if missing else v}")
    if s['non_default_args']:
        print(f"  非默认参数：{s['non_default_args']}")
    for e in s['errors'] + s['warnings']:
        print(f"  [!] {e}：{ADVICE[e]}")
    if ledger_cmp:
        print('\n  和账本对账（误差 = 日志 / 账本 - 1）')
        for name, log_v, led_v in ledger_cmp:
            if log_v is None:
                print(f'    {name:<14} 日志缺失     账本 {led_v:,.2f}')
                continue
            err = log_v / led_v - 1 if led_v else float('nan')
            flag = '' if abs(err) <= 0.10 else '  ← 超过 10%，先用日志里的激活峰值和 Graph 显存回填 --act-gib/--graph-gib'
            print(f'    {name:<14} 日志 {log_v:>12,.2f}  账本 {led_v:>12,.2f}  误差 {err:+.1%}{flag}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--model', help='给出时调用 02-ledger 的账本对账，取值同 sim_ledger.py')
    ap.add_argument('--gpu')
    ap.add_argument('--tp', type=int, default=1)
    ap.add_argument('--util', type=float, default=0.92)
    a = ap.parse_args()
    with open(a.log, encoding='utf-8', errors='replace') as f:
        s = summarize(*parse(f))
    if a.json:
        print(json.dumps(s, ensure_ascii=False, indent=2))
        return
    cmp_rows = None
    if a.model and a.gpu:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '02-ledger'))
        from common.specs import GPUS, MODELS  # noqa: E402
        from sim_ledger import ledger  # noqa: E402
        kw = dict(tp=a.tp, util=a.util)
        if s['graph_gib'] is not None:
            kw['graph_gib'] = s['graph_gib']
        L = ledger(MODELS[a.model], GPUS[a.gpu], **kw)
        cmp_rows = [('权重 GiB', s['weights_gib'], L['weights_gib']),
                    ('KV 可用 GiB', s['kv_avail_gib'], L['kv_gib']),
                    ('KV token', s['kv_tokens'], L['kv_tokens'])]
    print(f'{a.log}')
    show(s, cmp_rows)
    sys.exit(1 if s['errors'] else 0)


if __name__ == '__main__':
    main()
