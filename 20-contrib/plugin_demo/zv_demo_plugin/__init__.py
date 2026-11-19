"""zv_demo：一个最小的 vLLM general 插件。

entry point（见 setup.py）：
  vllm.general_plugins   zv_demo     = zv_demo_plugin:register
  vllm.endpoint_plugins  zv_demo_api = zv_demo_plugin.endpoints:ZvDemoEndpointPlugin

register() 会在每个加载插件的进程里各跑一次（vllm/plugins/__init__.py:77-90，plugins_loaded 只防同一进程重复），
spawn 模式下 TP=4 是 API server、模型探测子进程、EngineCore、4 个 worker，共 7 次。所以这里的每一步都必须：
  1. 幂等：同一进程内重复调用不报错（register_oot 遇到重名直接 assert，custom_op.py:338）
  2. 便宜：不要 import torch 重模块、不要碰 GPU，模型类用 "module:Class" 字符串懒加载
  3. 不抛异常：vLLM 不给 func() 包 try，这里抛错整个 vllm serve 起不来（SGLang 的 load_plugins 则只打日志）
"""
import logging
import os

logger = logging.getLogger(__name__)

ARCH = 'ZvLlamaForCausalLM'
LAZY_CLS = 'vllm.model_executor.models.llama:LlamaForCausalLM'


def _register_model():
    from vllm import ModelRegistry
    if ARCH in ModelRegistry.get_supported_archs():
        return False
    # 字符串形式：注册时不 import 模型模块，第一次实例化时才 import（与 tests/plugins/vllm_add_dummy_model 相同写法）
    ModelRegistry.register_model(ARCH, LAZY_CLS)
    return True


def _register_oot_op():
    """把内置 SiluAndMul 换成带计数的子类。全局生效，所以默认关闭，ZV_DEMO_OOT=1 才打开。"""
    if os.environ.get('ZV_DEMO_OOT') != '1':
        return False
    from vllm.model_executor.custom_op import CustomOp, op_registry_oot
    if 'SiluAndMul' in op_registry_oot:            # 别的插件或本进程已注册过：让路，不要撞 assert
        logger.warning('zv_demo: SiluAndMul already has an OOT override, skip')
        return False
    from vllm.model_executor.layers.activation import SiluAndMul

    @CustomOp.register_oot(name='SiluAndMul')
    class ZvSiluAndMul(SiluAndMul):
        calls = 0

        def forward_cuda(self, x):
            ZvSiluAndMul.calls += 1
            return super().forward_cuda(x)

    return True


def register():
    try:
        added = _register_model()
        oot = _register_oot_op()
    except Exception:                                # 自己兜住，别让 vllm serve 因为一个演示插件起不来
        logger.exception('zv_demo: register failed, plugin disabled in pid %d', os.getpid())
        return
    logger.info('zv_demo: register() pid=%d model=%s oot=%s', os.getpid(), added, oot)
