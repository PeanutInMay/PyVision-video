# NOTE: Env must be imported here in order to trigger metaclass registering.
# 这些 import 的主要目的不是直接调用类，而是触发 ToolMeta，把各工具按 name
# 注册到 ToolBase.registry。真正运行时，ParallelEnv 根据数据里的 env_name
# 动态创建工具。PyVision 当前主线使用 agents_x 下的 Python code 工具；
# 其它 env 多是 RAG、frozenlake、旧版 visual toolbox / visual agent 等实验路线。
from .envs.rag_engine.rag_engine import RAGEngineEnv
from .envs.rag_engine.rag_engine_v2 import RAGEngineEnvV2
from .envs.visual_agent.vl_agent_v1 import VLAgentEnvV1
from .envs.visual_agent.vl_agent_v2 import VLAgentEnvV2
from .envs.mm_process_engine.visual_toolbox import VisualToolBox
from .envs.mm_process_engine.visual_toolbox_v2 import VisualToolBoxV2
from .envs.mm_process_engine.visual_toolbox_v3 import VisualToolBoxV3
from .envs.mm_process_engine.visual_toolbox_v4 import VisualToolBoxV4
from .envs.mm_process_engine.visual_toolbox_v5 import VisualToolBoxV5
from .envs.visual_agent.vl_agent_v2 import VLAgentEnvV2
from .envs.visual_agent.vl_agent_v3 import VLAgentEnvV3
from .envs.agents_x.safe_persis_python_exe_tool_w_image_hint import MultiModalPythonTool_w_Image_Hint
from .envs.agents_x.safe_persis_python_exe_tool_wo_image_hint import MultiModalPythonTool_wo_Image_Hint
from .envs.agents_x.safe_persis_python_exe_tool_wo_video_hint import MultiModalPythonTool_wo_Video_Hint

try:
    from .envs.visual_agent.mm_search_engine import MMSearchEngine
except Exception as err:
    print(f' [ERROR] Failed to register MMSearchEngine : {err=}')

try:
    from .envs.frozenlake.frozenlake import FrozenLakeTool
except Exception as err:
    print(f' [ERROR] Failed to register FrozenLakeTool : {err=}')

from .parallel_env import agent_rollout_loop
