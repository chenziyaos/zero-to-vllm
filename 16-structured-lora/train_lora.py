"""自己训一个 LoRA adapter，产出 vLLM / SGLang 都能直接加载的 PEFT 目录。

    python3 train_lora.py                                           # 默认：随机权重的小 Llama，CPU 上几秒，只验证流程
    python3 train_lora.py --base Qwen/Qwen3-0.6B --steps 80 --out loras/city-r8
    python3 train_lora.py --base Qwen/Qwen3-0.6B --rank 64 --out loras/city-r64   # 换 rank，给 lora_mem / lora_bench 用

依赖 torch、transformers、peft（pip install -r requirements-lora.txt）。

数据是 data/city_country.jsonl：输入「城市：巴黎」，输出一行 JSON {"city": "巴黎", "country": "法国"}。
底座模型不会自发输出这个格式，训完的 adapter 会，所以加载前后的差别一眼就能看出来。
最后 4 条不参与训练，用来看训完之后在没见过的城市上能不能答对。

训练只对 completion 部分算 loss（prompt 部分的 label 设成 -100），优化器是 AdamW，不做梯度累积和调度。
保存的目录里只有 adapter_config.json 和 adapter_model.safetensors，交给 check_adapter.py 检查后再上线。
"""
import argparse
import json
import os
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
TINY = 'hf-internal-testing/tiny-random-LlamaForCausalLM'


def load_data(path, n_eval):
    rows = [json.loads(line) for line in open(path, encoding='utf-8') if line.strip()]
    return rows[:-n_eval], rows[-n_eval:]


def encode(tok, row):
    p = tok(row['prompt'], add_special_tokens=False)['input_ids']
    c = tok(row['completion'], add_special_tokens=False)['input_ids'] + [tok.eos_token_id]
    return p + c, [-100] * len(p) + c


def batch(tok, rows):
    enc = [encode(tok, r) for r in rows]
    n = max(len(i) for i, _ in enc)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.tensor([i + [pad] * (n - len(i)) for i, _ in enc])
    labels = torch.tensor([lab + [-100] * (n - len(lab)) for _, lab in enc])
    mask = torch.tensor([[1] * len(i) + [0] * (n - len(i)) for i, _ in enc])
    return ids, labels, mask


@torch.no_grad()
def answer(model, tok, prompt, max_new=32):
    ids = tok(prompt, return_tensors='pt', add_special_tokens=False)['input_ids']
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip().split('\n')[0]


def accuracy(model, tok, rows):
    hits = [answer(model, tok, r['prompt']) == r['completion'] for r in rows]
    return sum(hits) / len(hits)


@torch.no_grad()
def eval_loss(model, tok, rows):
    ids, labels, mask = batch(tok, rows)
    return model(input_ids=ids, attention_mask=mask, labels=labels).loss.item()


def train(base=TINY, data=os.path.join(HERE, 'data', 'city_country.jsonl'), out='loras/city-r8', rank=8, alpha=16,
          targets=('q_proj', 'k_proj', 'v_proj', 'o_proj'), steps=60, lr=2e-3, batch_size=8, n_eval=4, seed=0,
          log=print):
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    train_rows, eval_rows = load_data(data, n_eval)
    probe = eval_rows[0]['prompt']
    before = {'loss': eval_loss(model, tok, eval_rows), 'acc': accuracy(model, tok, eval_rows),
              'probe': answer(model, tok, probe)}

    cfg = LoraConfig(r=rank, lora_alpha=alpha, target_modules=list(targets), lora_dropout=0.0, bias='none',
                     task_type='CAUSAL_LM')
    model = get_peft_model(model, cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    log(f'底座 {base}，LoRA r={rank} alpha={alpha} 模块 {list(targets)}')
    log(f'可训练参数 {n_train:,} / 总参数 {n_all:,}（{n_train / n_all:.3%}）')

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    g = torch.Generator().manual_seed(seed)
    losses, t0 = [], time.perf_counter()
    model.train()
    for step in range(steps):
        idx = torch.randperm(len(train_rows), generator=g)[:batch_size].tolist()
        ids, labels, mask = batch(tok, [train_rows[i] for i in idx])
        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
        if step % max(1, steps // 6) == 0 or step == steps - 1:
            log(f'  step {step:4d}  loss {loss.item():.4f}')
    model.eval()
    after = {'loss': eval_loss(model, tok, eval_rows), 'acc': accuracy(model, tok, eval_rows),
             'probe': answer(model, tok, probe)}

    model.save_pretrained(out)
    log(f'训练 {steps} 步，用时 {time.perf_counter() - t0:.1f} s，adapter 存到 {out}')
    log(f'留出的 {n_eval} 条：loss {before["loss"]:.3f} → {after["loss"]:.3f}，'
        f'完全答对 {before["acc"]:.0%} → {after["acc"]:.0%}')
    log(f'探针 {probe.strip()!r}')
    log(f'  训练前 {before["probe"]!r}')
    log(f'  训练后 {after["probe"]!r}')
    return {'losses': losses, 'before': before, 'after': after, 'out': out, 'trainable': n_train}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--base', default=TINY, help=f'底座模型，默认 {TINY}（随机权重，只验证流程）')
    ap.add_argument('--data', default=os.path.join(HERE, 'data', 'city_country.jsonl'))
    ap.add_argument('--out', default='loras/city-r8')
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--alpha', type=int, default=16)
    ap.add_argument('--targets', nargs='+', default=['q_proj', 'k_proj', 'v_proj', 'o_proj'])
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(argv)
    train(a.base, a.data, a.out, a.rank, a.alpha, a.targets, a.steps, a.lr, a.batch_size, seed=a.seed)


if __name__ == '__main__':
    main()
