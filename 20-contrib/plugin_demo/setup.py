from setuptools import setup

setup(
    name='zv_demo_plugin',
    version='0.1',
    packages=['zv_demo_plugin'],
    entry_points={
        # 等号左边是插件名，VLLM_PLUGINS 按这个名字放行（不是包名）
        'vllm.general_plugins': ['zv_demo = zv_demo_plugin:register'],
        'vllm.endpoint_plugins': ['zv_demo_api = zv_demo_plugin.endpoints:ZvDemoEndpointPlugin'],
    },
)
