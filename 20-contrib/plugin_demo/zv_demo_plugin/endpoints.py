"""endpoint 插件：在 vllm serve 上加一条 GET /plugins/zv_demo/scheduler_config。

契约来自 vllm/plugins/endpoint_plugins/interface.py 与 docs/design/endpoint_plugins.md：
  - 阶段 A build_app()：attach_router(app)，此时没有 engine，只注册路由
  - 阶段 B init_app_state()：init_state(engine_client, state, args)；CPU-only render server 上 engine_client 是 None
  - 只有 VLLM_PLUGINS 显式列出 "zv_demo_api" 才加载（endpoint_plugins.md:107）
  - 路由用 /plugins/<name>/ 前缀：同路径同方法后注册的覆盖先注册的，冲突不报错（:129-134）
"""
from fastapi import FastAPI, HTTPException, Request

ROUTE = '/plugins/zv_demo/scheduler_config'


class ZvDemoEndpointPlugin:
    name = 'zv_demo_api'
    required_tasks = None            # None：render server 上也会加载，所以 handler 必须处理 engine_client=None

    def attach_router(self, app: FastAPI) -> None:
        @app.get(ROUTE)
        async def scheduler_config(raw_request: Request):
            client = getattr(raw_request.app.state, 'zv_engine_client', None)
            if client is None:
                raise HTTPException(status_code=503, detail='no engine on this server (render-only)')
            try:
                results = await client.collective_rpc('zv_get_scheduler_config')
            except NotImplementedError as e:
                # 没带 --worker-extension-cls 时，worker 侧报 "Method 'zv_get_scheduler_config' is not implemented."
                raise HTTPException(status_code=500, detail=str(e)) from e
            return {'workers': results}

    async def init_state(self, engine_client, state, args) -> None:
        state.zv_engine_client = engine_client
