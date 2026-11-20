"""上线前检查一个 PEFT LoRA adapter 能不能被 vLLM / SGLang 加载，并给出对应的启动参数。只用标准库。

    python3 check_adapter.py ADAPTER_DIR [ADAPTER_DIR ...]
    python3 check_adapter.py ADAPTER_DIR --max-lora-rank 16 --lora-target-modules q_proj v_proj

读 adapter_config.json 和 adapter_model.safetensors 的头部（不读权重本体），逐条对照两边加载时的校验：

vLLM（主干 0d6f92a）
  vllm/config/lora.py:27        max_lora_rank 只能取 1, 8, 16, 32, 64, 128, 256, 320, 512，默认 16（:34）
  vllm/lora/peft_helper.py:54   _validate_features：modules_to_save 只允许 classifier / score，不支持 DoRA、lora_bias、
                                会改底座权重的 init（PiSSA、OLoRA、CorDA、LoftQ）、aLoRA、layer_replication 等
  vllm/lora/peft_helper.py:195  validate_legal：max(r, rank_pattern) 不能超过 --max-lora-rank；bias 必须是 none
  vllm/lora/lora_model.py:250   safetensors 里出现模型不支持 LoRA 的模块名，直接报 unexpected modules

SGLang（主干 8fabf5e，路径相对 python/sglang/srt/）
  lora/lora_manager.py:337      validate_new_adapter：不支持往词表里加 token 的 adapter，不支持 DoRA；
                                rank 超过 --max-lora-rank、或模块不在 --lora-target-modules 里，报 incompatible
  lora/lora_manager.py:896      不传 --max-lora-rank 时取启动时所有 adapter 的最大 r，之后热加载更大 rank 的会失败
  lora/utils.py:277             target_modules 归一化：q/k/v_proj → qkv_proj，gate/up_proj → gate_up_proj
"""
import argparse
import json
import os
import struct
import sys

VLLM_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)
VLLM_INIT_OK = ('gaussian', 'eva', 'orthogonal', 'mica', 'lora_ga')
VLLM_UNSUPPORTED_FLAGS = {
    'use_dora': '不支持 DoRA',
    'lora_bias': '不支持 lora_bias',
    'alora_invocation_tokens': '不支持 aLoRA',
    'layer_replication': '不支持 layer_replication',
    'use_bdlora': '不支持 BD-LoRA',
    'use_qalora': '不支持 QALoRA',
}
SGL_NORMALIZE = {'q_proj': 'qkv_proj', 'k_proj': 'qkv_proj', 'v_proj': 'qkv_proj',
                 'gate_proj': 'gate_up_proj', 'up_proj': 'gate_up_proj'}


def read_safetensors_header(path):
    """safetensors 文件开头是 8 字节小端长度，后面跟一段 JSON 头，记录每个张量的 dtype 和 shape。"""
    with open(path, 'rb') as f:
        (n,) = struct.unpack('<Q', f.read(8))
        return json.loads(f.read(n))


def load(adapter_dir):
    cfg = json.load(open(os.path.join(adapter_dir, 'adapter_config.json')))
    st = os.path.join(adapter_dir, 'adapter_model.safetensors')
    tensors = {k: v for k, v in read_safetensors_header(st).items() if k != '__metadata__'} if os.path.exists(st) else {}
    added = os.path.join(adapter_dir, 'added_tokens.json')
    added_tokens = json.load(open(added)) if os.path.exists(added) else {}
    return cfg, tensors, added_tokens


def lora_modules(tensors):
    """从张量名里取出真正带 LoRA 权重的模块短名，例如 ...layers.0.self_attn.q_proj.lora_A.weight → q_proj。"""
    mods = set()
    for k in tensors:
        parts = k.split('.')
        for tag in ('lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B'):
            if tag in parts:
                mods.add(parts[parts.index(tag) - 1])
    return mods


def max_rank(cfg):
    return max([cfg['r'], *(cfg.get('rank_pattern') or {}).values()])


def vllm_rank_flag(r):
    for x in VLLM_RANKS:
        if x >= r:
            return x
    return None


def check_vllm(cfg, max_lora_rank=None):
    errs = []
    save = [m for m in (cfg.get('modules_to_save') or []) if m not in ('classifier', 'score')]
    if save:
        errs.append(f'modules_to_save 只允许 classifier / score，这里有 {save}')
    for k, msg in VLLM_UNSUPPORTED_FLAGS.items():
        if cfg.get(k):
            errs.append(msg)
    init = cfg.get('init_lora_weights', True)
    if isinstance(init, str) and init.lower() not in VLLM_INIT_OK:
        errs.append(f'init_lora_weights={init!r} 会改底座权重，需先用 PEFT 转成普通 LoRA')
    if cfg.get('bias', 'none') != 'none':
        errs.append(f'bias={cfg["bias"]!r}，只支持 none')
    r = max_rank(cfg)
    if max_lora_rank is not None:
        if max_lora_rank not in VLLM_RANKS:
            errs.append(f'--max-lora-rank {max_lora_rank} 不在可选值 {VLLM_RANKS} 里')
        elif r > max_lora_rank:
            errs.append(f'rank {r} 大于 --max-lora-rank {max_lora_rank}')
    elif vllm_rank_flag(r) is None:
        errs.append(f'rank {r} 超过 vLLM 的上限 {VLLM_RANKS[-1]}')
    return errs


def sgl_targets(names):
    return {SGL_NORMALIZE.get(n, n) for n in names}


def check_sglang(cfg, tensors, added_tokens, max_lora_rank=None, target_modules=None):
    errs = []
    if added_tokens:
        errs.append(f'adapter 往词表里加了 {len(added_tokens)} 个 token，SGLang 不支持（底座词表里已有的 id 会被忽略，见 lora_config.py）')
    if cfg.get('use_dora'):
        errs.append('不支持 DoRA')
    r = cfg['r']
    if max_lora_rank is not None and r > max_lora_rank:
        errs.append(f'rank {r} 大于 --max-lora-rank {max_lora_rank}')
    if target_modules is not None:
        have = sgl_targets(lora_modules(tensors) or cfg.get('target_modules') or [])
        allowed = {'all'} if 'all' in target_modules or 'all-linear' in target_modules else sgl_targets(target_modules)
        miss = sorted(have - allowed) if 'all' not in allowed else []
        if miss:
            errs.append(f'模块 {miss} 不在 --lora-target-modules 里')
    return errs


def report(adapter_dir, args, out=None):
    out = out or sys.stdout
    cfg, tensors, added = load(adapter_dir)
    name = os.path.basename(os.path.normpath(adapter_dir))
    path = os.path.abspath(adapter_dir)
    mods = sorted(lora_modules(tensors))
    declared = cfg.get('target_modules')
    r = max_rank(cfg)
    print(f'== {name}', file=out)
    print(f'  底座      {cfg.get("base_model_name_or_path")}', file=out)
    print(f'  r / alpha {cfg["r"]} / {cfg.get("lora_alpha")}'
          + (f'，rank_pattern 最大 {r}' if cfg.get('rank_pattern') else '')
          + ('，rsLoRA' if cfg.get('use_rslora') else ''), file=out)
    print(f'  声明模块  {declared}', file=out)
    print(f'  实际模块  {mods}（{len(tensors)} 个张量）', file=out)
    if isinstance(declared, list) and set(declared) - set(mods) and tensors:
        print(f'  注意：声明了 {sorted(set(declared) - set(mods))} 但 safetensors 里没有，底座里多半没有同名层', file=out)
    ve = check_vllm(cfg, args.max_lora_rank)
    se = check_sglang(cfg, tensors, added, args.max_lora_rank, args.lora_target_modules)
    for who, errs in (('vLLM', ve), ('SGLang', se)):
        print(f'  {who:7} {"可以加载" if not errs else "不能加载：" + "；".join(errs)}', file=out)
    rk = args.max_lora_rank or vllm_rank_flag(r)
    if not ve:
        print(f'  vLLM    --enable-lora --max-lora-rank {rk} --lora-modules {name}={path}', file=out)
    if not se:
        tm = ' '.join(mods or declared or [])
        print(f'  SGLang  --enable-lora --max-lora-rank {args.max_lora_rank or r} --lora-target-modules {tm}'
              f' --lora-paths {name}={path}', file=out)
    return not ve and not se


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('adapters', nargs='+')
    ap.add_argument('--max-lora-rank', type=int, help='打算用的 --max-lora-rank；不填则按 adapter 推荐')
    ap.add_argument('--lora-target-modules', nargs='+', help='SGLang 打算用的 --lora-target-modules')
    a = ap.parse_args(argv)
    ok = [report(d, a) for d in a.adapters]
    return 0 if all(ok) else 1


if __name__ == '__main__':
    sys.exit(main())
