#!/usr/bin/env python3
"""生成部署和压测命令；只打印，绝不启动服务。"""
import argparse
import json
import shlex

PROFILES = {
    '5090-nvfp4': ('Inferact/Qwen3.8-27B-NVFP4', 1, 32768, True),
    '5090x2-fp8': ('Qwen/Qwen3.8-27B-FP8', 2, 262144, False),
    '5090x2-nvfp4': ('Inferact/Qwen3.8-27B-NVFP4', 2, 262144, False),
    'h100-fp8': ('Qwen/Qwen3.8-27B-FP8', 1, 32768, False),
    'h100-int4': ('RedHatAI/Qwen3.8-27B-INT4', 1, 32768, False),
    'b200-mxfp4': ('RedHatAI/Qwen3.8-27B-MXFP4', 1, 262144, False),
    'h200-bf16': ('Qwen/Qwen3.8-27B', 1, 262144, False),
}


def command(profile, engine='vllm', mtp=None, text_only=False):
    model, tp, length, eager = PROFILES[profile]
    if mtp is None:
        mtp = 3 if profile in ('5090x2-fp8', '5090x2-nvfp4') else 0
    if mtp < 0:
        raise ValueError('草稿数不能为负数')
    if engine == 'sglang':
        if profile != 'h200-bf16' or mtp or text_only:
            raise ValueError('SGLang 只提供 BF16 对照起点；其他组合尚未做兼容验证')
        return ['python3', '-m', 'sglang.launch_server', '--model-path', model,
                '--tp-size', str(tp), '--context-length', str(length),
                '--host', '127.0.0.1', '--port', '30000',
                '--reasoning-parser', 'qwen3', '--tool-call-parser', 'qwen3_coder']
    cmd = ['vllm', 'serve', model, '--served-model-name', 'qwen38',
           '--tensor-parallel-size', str(tp), '--max-model-len', str(length),
           '--max-num-seqs', '8', '--gpu-memory-utilization', '0.90',
           '--kv-cache-dtype', 'fp8', '--host', '127.0.0.1', '--port', '8000',
           '--reasoning-parser', 'qwen3', '--enable-auto-tool-choice',
           '--tool-call-parser', 'qwen3_coder']
    if eager:
        cmd.append('--enforce-eager')
    if text_only:
        cmd.append('--language-model-only')
    if mtp:
        cmd += ['--speculative-config', json.dumps({'method': 'mtp', 'num_speculative_tokens': mtp}, separators=(',', ':'))]
    return cmd


def bench(base_url, model, input_len=1024, output_len=256, concurrency=1, prompts=64, rate='inf'):
    if min(input_len, output_len, concurrency, prompts) <= 0:
        raise ValueError('长度、并发、请求数必须为正数')
    if not base_url.startswith(('http://','https://')):
        raise ValueError('base-url 必须是 HTTP(S) 地址')
    return ['vllm', 'bench', 'serve', '--backend', 'openai-chat',
            '--base-url', base_url.rstrip('/'), '--endpoint', '/v1/chat/completions',
            '--model', model, '--tokenizer', 'Qwen/Qwen3.8-27B',
            '--dataset-name', 'random', '--random-input-len', str(input_len),
            '--random-output-len', str(output_len), '--num-prompts', str(prompts),
            '--max-concurrency', str(concurrency), '--request-rate', str(rate),
            '--seed', '42', '--save-result', '--result-dir', 'results',
            '--result-filename', f'bench-i{input_len}-o{output_len}-c{concurrency}.json',
            '--percentile-metrics', 'ttft,tpot,itl,e2el', '--metric-percentiles', '50,95,99',
            '--extra-body', json.dumps({'temperature':0,'chat_template_kwargs':{'enable_thinking':False}})]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile', choices=PROFILES, default='5090-nvfp4')
    ap.add_argument('--engine', choices=['vllm','sglang'], default='vllm')
    ap.add_argument('--mtp', type=int, default=None, help='覆盖档位草稿数；0 表示关闭')
    ap.add_argument('--text-only', action='store_true')
    ap.add_argument('--bench', action='store_true')
    ap.add_argument('--base-url', default='http://127.0.0.1:8000')
    ap.add_argument('--model', default='qwen38')
    ap.add_argument('--input-len', type=int, default=1024)
    ap.add_argument('--output-len', type=int, default=256)
    ap.add_argument('--concurrency', type=int, default=1)
    ap.add_argument('--prompts', type=int, default=64)
    a=ap.parse_args()
    cmd = bench(a.base_url,a.model,a.input_len,a.output_len,a.concurrency,a.prompts) if a.bench else command(a.profile,a.engine,a.mtp,a.text_only)
    print('# 仅生成命令；GPU 环境运行前核对模型 revision、镜像 digest 和 --help。')
    print(shlex.join(cmd))

if __name__=='__main__':
    main()
