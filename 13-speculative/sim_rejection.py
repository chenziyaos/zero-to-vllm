"""拒绝采样 Monte Carlo：哪些验证规则无损、哪些有损、各自多接受多少（只依赖标准库）。

对照的源码（vllm 主干 0d6f92a，sglang 主干 8fabf5e）：
  vllm   v1/worker/gpu/spec_decode/rejection_sampler_utils.py:520–790   _rejection_kernel（MRV2）
           standard：log p(x) > log u + log q(x)，拒绝后从 max(p - q, 0) 重采
           block   ：p_i = min(p_(i-1)·p(x_i)/q(x_i), 1)；h_i = res/(res + 1 - p_i)，最后一位 h = p_k；
                     τ = max{i : u_i ≤ h_i}（每位独立的 u）；拒绝后从 max(p_τ·p - q, 0) 重采
           synthetic：u < 条件接受率，和 p、q 无关
           fly     ：原生拒绝 + 熵 ≥ 门限 + 后面还有 W 位 → 挂起；后面 W 次原生接受全过才把挂起的那位也收下
         v1/spec_decode/fly.py:16                compute_fly_entropy（top-k 熵）
         v1/sample/rejection_sampler.py:745      sample_recovered_tokens：argmax(r(x) / E_x)，E ~ Exp(1)
  sglang jit_kernel/.../speculative/sampling.cuh  TreeSpeculativeSamplingTargetOnly（topk=1 链的情形）
           接受：p > 0 且 (coin < p / threshold_acc 或 p ≥ threshold_single)
           拒绝：draft_probs[x] = p(x)，最后从 relu(target - draft_probs) 采

模型：词表 V=4 的一阶马尔可夫链。目标 p(·|prev) 与草稿 q(·|prev) 各是一张 4×4 转移表，
草稿 = 目标的 logits 加噪声。无损的定义：生成序列前 3 个 token 的联合分布（64 种结果）和直接从 p 采样完全一样。
TV 距离用来量化偏差；"噪声底" 是直接从 p 采同样多次的 TV（有限样本带来的）。

用法：
  python3 sim_rejection.py --scenario lossless    # 各规则的 TV 和每轮产出 token 数
  python3 sim_rejection.py --scenario recovered   # 指数噪声 argmax 等价于按 max(p-q,0) 归一化采样
  python3 sim_rejection.py --scenario greedy      # 草稿 greedy（one-hot） vs probabilistic：谁的接受率高
  python3 sim_rejection.py --scenario threshold   # SGLang threshold_acc / threshold_single 扫描
  python3 sim_rejection.py --scenario fly         # FLy 熵门限、窗口扫描
"""
import argparse
import math
import random

V = 4
K = 3          # 草稿 token 数
L = 3          # 只看生成序列的前 L 个 token


# --------------------------------------------------------------------------- 分布
def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [x / s for x in e]


def make_models(seed=7, noise=1.0, sharp=1.5):
    rng = random.Random(seed)
    target, draft = [], []
    for _ in range(V):
        logits = [rng.gauss(0, sharp) for _ in range(V)]
        target.append(softmax(logits))
        draft.append(softmax([x + rng.gauss(0, noise) for x in logits]))
    return target, draft


def sample(rng, probs):
    u = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if u < acc:
            return i
    return len(probs) - 1


def norm_relu(p, q):
    r = [max(a - b, 0.0) for a, b in zip(p, q)]
    s = sum(r)
    if s <= 0:
        return list(p)
    return [x / s for x in r]


def one_hot(i):
    return [1.0 if j == i else 0.0 for j in range(V)]


def argmax(xs):
    return max(range(len(xs)), key=lambda i: xs[i])


def topk_entropy(p, k=3):
    top = sorted(p, reverse=True)[:k]
    return -sum(x * math.log(x) for x in top if x > 0)


def exact_joint(target, start=0):
    joint = {}

    def rec(prev, prefix, prob):
        if len(prefix) == L:
            joint[tuple(prefix)] = prob
            return
        for x in range(V):
            rec(x, prefix + [x], prob * target[prev][x])
    rec(start, [], 1.0)
    return joint


def tv(counts, joint, n):
    return 0.5 * sum(abs(counts.get(key, 0) / n - p) for key, p in joint.items())


# --------------------------------------------------------------------------- 一轮投机：返回本轮产出的 token
def draft_chain(rng, draft, prev, greedy_draft):
    xs, qs = [], []
    for _ in range(K):
        q = draft[prev]
        x = argmax(q) if greedy_draft else sample(rng, q)
        xs.append(x)
        qs.append(one_hot(x) if greedy_draft else q)
        prev = x
    return xs, qs


def ctx_of(prev, xs, i):
    """第 i 个草稿位置（0 起）的条件上文 = 前一个 token。"""
    return prev if i == 0 else xs[i - 1]


def round_standard(rng, target, draft, prev, greedy_draft=False):
    """Leviathan / Chen：逐位比值检验。vLLM V1 与 MRV2 的 standard 都是它。"""
    xs, qs = draft_chain(rng, draft, prev, greedy_draft)
    out = []
    for i in range(K):
        p = target[ctx_of(prev, xs, i)]
        x = xs[i]
        if rng.random() * qs[i][x] < p[x]:          # p(x) > u·q(x)
            out.append(x)
        else:
            out.append(sample(rng, norm_relu(p, qs[i])))
            return out
    out.append(sample(rng, target[xs[-1]]))
    return out


def round_block(rng, target, draft, prev, greedy_draft=False):
    """Sun et al. 2024 块验证，按 vLLM _rejection_kernel 的 USE_BLOCK_VERIFICATION 分支写。"""
    xs, qs = draft_chain(rng, draft, prev, greedy_draft)
    ps = [target[ctx_of(prev, xs, i)] for i in range(K)]
    # cum[i] = p_(i+1)
    cum, lp = [], 1.0
    for i in range(K):
        lp = min(lp * ps[i][xs[i]] / qs[i][xs[i]], 1.0) if qs[i][xs[i]] > 0 else 0.0
        cum.append(lp)
    accepted = 0
    for i in range(K):
        u = rng.random()
        if i < K - 1:
            pn, qn = ps[i + 1], qs[i + 1]
            res = sum(max(cum[i] * a - b, 0.0) for a, b in zip(pn, qn))
            denom = res + 1.0 - cum[i]
            h = res / denom if denom > 0 else 1.0
        else:
            h = cum[i]
        if u <= h:
            accepted = i + 1
    out = xs[:accepted]
    if accepted == K:
        out.append(sample(rng, target[xs[-1]]))
    else:
        p_tau = 1.0 if accepted == 0 else cum[accepted - 1]
        p, q = ps[accepted], qs[accepted]
        out.append(sample(rng, norm_relu([p_tau * a for a in p], q)))
    return out


def round_synthetic(rng, target, draft, prev, rates, greedy_draft=False):
    """synthetic：第 i 位以条件率接受，跟 p/q 无关；拒绝后照常从残差采。"""
    xs, qs = draft_chain(rng, draft, prev, greedy_draft)
    out = []
    for i in range(K):
        p = target[ctx_of(prev, xs, i)]
        if rng.random() < rates[i]:
            out.append(xs[i])
        else:
            out.append(sample(rng, norm_relu(p, qs[i])))
            return out
    out.append(sample(rng, target[xs[-1]]))
    return out


def round_fly(rng, target, draft, prev, window, threshold, top_k=3, greedy_draft=False):
    """FLy：按 _rejection_kernel 的 FLY_WINDOW_SIZE > 0 分支（非 greedy 请求）逐行写。"""
    xs, qs = draft_chain(rng, draft, prev, greedy_draft)
    ps = [target[ctx_of(prev, xs, i)] for i in range(K)]
    accepted_length, pending, verifying = 0, -1, True
    for i in range(K):
        if not verifying:
            break
        x = xs[i]
        accepted = rng.random() * qs[i][x] < ps[i][x]
        can_defer = ps[i][x] > 0 and qs[i][x] > 0
        if pending >= 0:
            verifying = accepted
            if accepted and i - pending == window:
                accepted_length = i + 1
                pending = -1
        elif accepted:
            accepted_length = i + 1
        else:
            verifying = can_defer and i + window < K and topk_entropy(ps[i], top_k) >= threshold
            if verifying:
                pending = i
    out = xs[:accepted_length]
    if accepted_length == K:
        out.append(sample(rng, target[xs[-1]]))
    else:
        out.append(sample(rng, norm_relu(ps[accepted_length], qs[accepted_length])))
    return out


def round_sgl_target_only(rng, target, draft, prev, th_single=1.0, th_acc=1.0):
    """SGLang TreeSpeculativeSamplingTargetOnly，topk=1 的链。草稿概率不参与（draft_probs 初始全 0）。"""
    xs, _ = draft_chain(rng, draft, prev, greedy_draft=True)
    out = []
    coin = rng.random()
    prob_acc = 0.0
    for i in range(K):
        p = target[ctx_of(prev, xs, i)]
        x = xs[i]
        prob_acc += p[x]
        if p[x] > 0 and (coin < prob_acc / th_acc or p[x] >= th_single):
            out.append(x)
            prob_acc = 0.0
            coin = rng.random()
        else:
            dp = [0.0] * V
            dp[x] = p[x]                      # "FIXME: leverage draft probs"
            out.append(sample(rng, norm_relu(p, dp)))
            return out
    out.append(sample(rng, target[xs[-1]]))
    return out


# --------------------------------------------------------------------------- 驱动
def run(round_fn, target, n, seed):
    rng = random.Random(seed)
    counts = {}
    rounds = tokens = 0
    for _ in range(n):
        seq, prev = [], 0
        while len(seq) < L:
            out = round_fn(rng, prev)
            rounds += 1
            tokens += len(out)
            seq += out
            prev = seq[-1]
        key = tuple(seq[:L])
        counts[key] = counts.get(key, 0) + 1
    return counts, tokens / rounds


def direct(target, n, seed):
    rng = random.Random(seed)
    counts = {}
    for _ in range(n):
        prev, seq = 0, []
        for _ in range(L):
            prev = sample(rng, target[prev])
            seq.append(prev)
        key = tuple(seq)
        counts[key] = counts.get(key, 0) + 1
    return counts


def alpha_stats(target, draft):
    """平稳意义下粗看一眼：每个上下文的 Σmin(p,q)（probabilistic 接受率）与 p(argmax q)（greedy 接受率）。"""
    prob = [sum(min(a, b) for a, b in zip(target[c], draft[c])) for c in range(V)]
    greedy = [target[c][argmax(draft[c])] for c in range(V)]
    return prob, greedy


def scenario_lossless(n):
    target, draft = make_models()
    joint = exact_joint(target)
    floor = tv(direct(target, n, 1), joint, n)
    prob, greedy = alpha_stats(target, draft)
    print(f'== V={V} 马尔可夫链，K={K} 个草稿，看前 {L} 个 token 的联合分布（{V ** L} 种结果），每种规则 {n} 条序列')
    print('  每个上下文 probabilistic 接受率 Σmin(p,q)：' + ', '.join(f'{x:.2f}' for x in prob))
    print('  每个上下文 greedy 草稿接受率 p(argmax q)： ' + ', '.join(f'{x:.2f}' for x in greedy))
    print(f'  噪声底（直接从 p 采 {n} 次的 TV）：{floor:.4f}\n')
    rates = [0.9, 0.8, 0.7]
    rows = [
        ('vLLM standard，草稿 probabilistic', lambda r, pv: round_standard(r, target, draft, pv), '无损'),
        ('vLLM standard，草稿 greedy（默认）', lambda r, pv: round_standard(r, target, draft, pv, True), '无损'),
        ('vLLM block，草稿 probabilistic', lambda r, pv: round_block(r, target, draft, pv), '无损'),
        ('vLLM block，草稿 greedy', lambda r, pv: round_block(r, target, draft, pv, True), '无损'),
        ('vLLM synthetic [0.9,0.8,0.7]', lambda r, pv: round_synthetic(r, target, draft, pv, rates), '有损'),
        ('vLLM fly W=1 熵≥0.3', lambda r, pv: round_fly(r, target, draft, pv, 1, 0.3), '有损'),
        ('SGLang target-only 阈值 1/1（默认）', lambda r, pv: round_sgl_target_only(r, target, draft, pv), '无损'),
        ('SGLang target-only acc=0.5', lambda r, pv: round_sgl_target_only(r, target, draft, pv, 1.0, 0.5), '有损'),
        ('SGLang target-only single=0.3', lambda r, pv: round_sgl_target_only(r, target, draft, pv, 0.3, 1.0), '有损'),
    ]
    print(f'  {"规则":<36}{"TV":<10}{"TV/噪声底":<12}{"每轮 token":<12}理论')
    results = {}
    for i, (name, fn, kind) in enumerate(rows):
        counts, per_round = run(fn, target, n, 100 + i)
        d = tv(counts, joint, n)
        results[name] = (d, per_round)
        print(f'  {name:<36}{d:<10.4f}{d / floor:<12.1f}{per_round:<12.3f}{kind}')
    print('\n  读法：TV 落在噪声底 1~2 倍以内 = 和直接从目标模型采样分不出来；有损规则的 TV 是噪声底的好几倍。')
    b_p = results['vLLM block，草稿 probabilistic'][1] / results['vLLM standard，草稿 probabilistic'][1] - 1
    b_g = results['vLLM block，草稿 greedy'][1] / results['vLLM standard，草稿 greedy（默认）'][1] - 1
    print(f'  block 比 standard 每轮多产出：草稿 probabilistic {b_p * 100:+.1f}%，草稿 greedy {b_g * 100:+.1f}%。')
    print('  probabilistic 时的增益来自 "整块一起判"：前面某位单看被拒，联合比值 Πp/Πq 还可能把整段捞回来。')
    print('  greedy 时几乎没有增益：草稿是确定的，前 i 位全收的概率上限就是 Πp(x_j)，逐位检验已经顶到这个上限。')
    return results, floor


def scenario_recovered(n):
    rng = random.Random(3)
    p = [0.40, 0.30, 0.20, 0.10]
    q = [0.10, 0.20, 0.10, 0.60]
    r = norm_relu(p, q)
    counts = [0] * V
    for _ in range(n):
        e = [rng.expovariate(1.0) for _ in range(V)]
        rel = [max(a - b, 0.0) for a, b in zip(p, q)]
        counts[argmax([rel[j] / e[j] for j in range(V)])] += 1
    print('== v1/sample/rejection_sampler.py sample_recovered_tokens：argmax(max(p-q,0) · inv_q)，inv_q = 1/Exp(1)')
    print(f'  p = {p}  q = {q}')
    print('  归一化残差 r = ' + ', '.join(f'{x:.3f}' for x in r))
    print('  指数噪声 argmax 频率 = ' + ', '.join(f'{c / n:.3f}' for c in counts) + f'（{n} 次）')
    print('  原理：x 胜出的概率 = r_x / Σr（指数分布的 "最小者" 性质，等价于 Gumbel-max）。')
    print('  好处：不用做前缀和，也不用归一化，一次 argmax 归约就采完；每个请求只抽一组噪声（"Create only one distribution')
    print('  for each request"），因为一个请求一轮最多只重采一个位置。')
    print('  草稿 greedy 时 q 是 one-hot：残差 = p 把草稿那个 token 挖掉，再重新归一化。')


def scenario_greedy(n):
    print('== 草稿 greedy（vLLM 默认 draft_sample_method）vs probabilistic：接受率谁高取决于两个分布长什么样')
    print('  probabilistic：接受率 = Σ min(p, q) = 1 - TV(p, q)；greedy：接受率 = p(argmax q)')
    print(f'  {"草稿噪声":<10}{"目标锐度":<10}{"prob 接受":<12}{"greedy 接受":<12}{"实测每轮 prob":<16}{"实测每轮 greedy":<16}')
    for noise, sharp in ((0.3, 3.0), (0.3, 1.0), (1.0, 3.0), (1.0, 1.0), (2.0, 1.5)):
        target, draft = make_models(seed=11, noise=noise, sharp=sharp)
        prob, greedy = alpha_stats(target, draft)
        _, pr = run(lambda r, pv: round_standard(r, target, draft, pv), target, n, 5)
        _, gr = run(lambda r, pv: round_standard(r, target, draft, pv, True), target, n, 6)
        print(f'  {noise:<10}{sharp:<10}{sum(prob) / V:<12.3f}{sum(greedy) / V:<12.3f}{pr:<16.3f}{gr:<16.3f}')
    print('  目标分布越尖（低温、事实性回答），greedy 草稿越接近 probabilistic；目标越平（高温、创作），')
    print('  probabilistic 的 Σmin(p,q) 明显更高——这就是 MRV2 要缓存草稿 logits（draft_sample_method=probabilistic）的理由。')
    print('  代价：每个草稿位置存一份 [vocab] 的 logits，128k 词表 × K 位 × max_num_seqs，这是实打实的显存。')


def scenario_threshold(n):
    target, draft = make_models()
    joint = exact_joint(target)
    floor = tv(direct(target, n, 1), joint, n)
    print('== SGLang --speculative-accept-threshold-single / --speculative-accept-threshold-acc 扫描（默认都是 1.0）')
    print(f'  噪声底 {floor:.4f}')
    print(f'  {"single":<8}{"acc":<8}{"TV":<10}{"每轮 token":<12}')
    for s, a in ((1.0, 1.0), (1.0, 0.8), (1.0, 0.5), (1.0, 0.3), (0.5, 1.0), (0.3, 1.0), (0.3, 0.5)):
        counts, pr = run(lambda r, pv: round_sgl_target_only(r, target, draft, pv, s, a), target, n, 21)
        print(f'  {s:<8}{a:<8}{tv(counts, joint, n):<10.4f}{pr:<12.3f}')
    print('  single 0.5 与 0.3 结果相同：这张表里草稿 argmax 位置的目标概率（0.17/0.16/0.29/0.78）没有落在 [0.3, 0.5)。')
    print('  acc 把接受概率从 p 抬到 min(1, p/acc)；single 让 p ≥ 阈值的草稿无条件通过。两者都只看目标概率，')
    print('  等于用 "目标觉得还行" 代替 "和目标同分布"——接受长度涨了，分布也偏了。')


def scenario_fly(n):
    target, draft = make_models()
    joint = exact_joint(target)
    floor = tv(direct(target, n, 1), joint, n)
    print('== vLLM FLy：窗口 W、熵门限扫描（K=3，所以 W ≤ 2；默认 W = min(6, K-1) = 2，门限 0.3，top-k 3）')
    print(f'  噪声底 {floor:.4f}')
    _, base = run(lambda r, pv: round_standard(r, target, draft, pv), target, n, 31)
    print(f'  standard 每轮 {base:.3f} token')
    print(f'  {"W":<4}{"熵门限":<10}{"TV":<10}{"每轮 token":<12}')
    for w, th in ((1, 99.0), (1, 1.0), (1, 0.3), (1, 0.0), (2, 0.3), (2, 0.0)):
        counts, pr = run(lambda r, pv: round_fly(r, target, draft, pv, w, th), target, n, 32)
        print(f'  {w:<4}{th:<10}{tv(counts, joint, n):<10.4f}{pr:<12.3f}')
    print('  门限设成 99 时没有位置能挂起，退化成 standard（无损）。门限越低、窗口越短，越多被拒 token 被 "后面都对" 捞回来，')
    print('  TV 随之变大。FLy 的赌注是：目标分布平坦（高熵）的位置上，草稿选了别的词，往往只是同义改写。')
    print('  W 越大，确认越严格（要后面连对 W 个）——K 小时 W 只能 ≤ K-1，所以 K=3 时 FLy 能捞回的很有限。')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='lossless',
                    choices=['lossless', 'recovered', 'greedy', 'threshold', 'fly'])
    ap.add_argument('-n', type=int, default=60000)
    a = ap.parse_args()
    {'lossless': scenario_lossless, 'recovered': scenario_recovered, 'greedy': scenario_greedy,
     'threshold': scenario_threshold, 'fly': scenario_fly}[a.scenario](a.n)


if __name__ == '__main__':
    main()
