#!/usr/bin/env python3
"""校验冒烟响应与 MTP 计数器；只处理输入文件，不调用模型。"""
import argparse
import json
import re
import sys
from pathlib import Path


def check_smoke(payload, kind):
    message = payload['choices'][0]['message']
    if kind == 'arithmetic':
        if (message.get('content') or '').strip() != '323':
            raise ValueError('算术冒烟失败：响应必须为 323')
    else:
        calls = message.get('tool_calls') or []
        valid = False
        for call in calls:
            fn = call.get('function', {})
            if fn.get('name') == 'get_weather':
                args = json.loads(fn['arguments'])
                city = args.get('city')
                valid = isinstance(city, str) and city.strip().lower() in ('北京', '北京市', 'beijing')
                if valid:
                    break
        if not valid:
            raise ValueError('工具冒烟失败：缺少 get_weather 调用或有效 city 参数')
    return True


def counters(text):
    pattern = re.compile(r'^(vllm:spec_decode_num_(?:accepted|draft)_tokens_total(?:\{[^\n]*\})?)\s+([\d.eE+\-]+)(?:\s+\d+)?$')
    result = {}
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            result[match[1]] = float(match[2])
    return result


def acceptance(before, after):
    start, end = counters(before), counters(after)
    if not start or start.keys() != end.keys():
        raise ValueError('指标缺失或标签集合变化；先预热并确认 MTP 开启，再采集同一服务的两个快照')
    total = {'accepted': 0.0, 'draft': 0.0}
    for key in start:
        delta = end[key] - start[key]
        if delta < 0:
            raise ValueError('计数器回退：服务可能重启，请重采快照')
        kind = 'accepted' if 'num_accepted_' in key else 'draft'
        total[kind] += delta
    if total['draft'] <= 0:
        raise ValueError('采样窗口没有草稿 token，不能计算接受率')
    if total['accepted'] > total['draft']:
        raise ValueError('accepted 大于 draft，请核验当前版本的指标语义')
    return {**total, 'acceptance': total['accepted'] / total['draft']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['arithmetic', 'tool', 'mtp'])
    parser.add_argument('files', nargs='*')
    args = parser.parse_args()
    if args.mode == 'mtp':
        if len(args.files) != 2:
            parser.error('mtp 需要前后两个 metrics 文件')
        print(json.dumps(acceptance(*(Path(p).read_text() for p in args.files)), ensure_ascii=False))
    else:
        check_smoke(json.load(sys.stdin), args.mode)
        print(f'{args.mode} 冒烟通过')


if __name__ == '__main__':
    main()
