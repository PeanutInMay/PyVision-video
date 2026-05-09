# PyVision-RL 训练部分理解笔记

本文档记录我目前对本项目训练部分的代码理解。重点是 PyVision-RL 的 agent 训练主链路：数据如何变成 prompt，多轮工具调用如何接入 vLLM rollout，工具 observation 如何拼回模型上下文，reward 如何计算，以及训练脚本中哪些参数真正控制行为。

## 1. 项目训练目标

这个项目不是普通 VLM SFT，也不是普通单轮 RLHF。它的核心目标是训练一个“会持续使用视觉工具”的多模态 agent。

README 中的描述是：多模态 RL agent 容易出现 interaction collapse，也就是训练后模型越来越少使用工具、多轮推理退化。PyVision-RL 的解决思路包括：

- 使用多轮 agent rollout，让模型在生成中可以写 Python code 调工具。
- 工具可以返回文本和图片 observation，再进入下一轮模型上下文。
- 使用 oversampling、filtering、ranking 选择更有训练价值的轨迹。
- 使用累积工具奖励，但只在答案正确时让工具奖励生效，避免模型为了奖励乱调用工具。
- 图像任务和视频任务共用一套 agent 训练框架，但视频任务强调按需读取视频帧，降低初始视觉 token 开销。

## 2. 训练入口

最外层训练入口主要是：

```text
verl_agents/run_train.sh
```

这个脚本是 Slurm 任务脚本，配置 1 节点 8 卡，然后调用：

```text
verl_agents/examples/agent/final_merged_v1v8_thinklite_single_node.sh
```

仓库里还有很多历史实验脚本，例如：

```text
verl_agents/examples/agent/train_pyvision_rl_7b_v1.sh
verl_agents/examples/agent/train_pyvision_rl_7b_v2.sh
verl_agents/examples/agent/train_pyvision_rl_7b_v3.sh
verl_agents/examples/agent/train_pyvision_rl_7b_v4.sh
verl_agents/examples/agent/final_merged_v1v8_thinklite.sh
verl_agents/examples/agent/final_merged_v1v8_thinklite_32b.sh
```

它们总体都调用同一个 Hydra 训练入口：

```bash
python3 -m verl.trainer.main_ppo
```

训练主配置来自：

```text
verl_agents/verl/trainer/config/ppo_trainer.yaml
```

训练脚本用命令行覆盖大量 Hydra 参数，例如：

- `data.train_files`
- `data.val_files`
- `data.train_batch_size`
- `data.max_prompt_length`
- `data.max_response_length`
- `data.return_raw_chat=True`
- `data.with_mm_hint=True/False`
- `actor_rollout_ref.rollout.name=vllm`
- `actor_rollout_ref.rollout.n=8/16`
- `actor_rollout_ref.rollout.agent.activate_agent=True`
- `actor_rollout_ref.rollout.agent.max_turns`
- `actor_rollout_ref.rollout.agent.single_response_max_tokens`
- `actor_rollout_ref.rollout.agent.tool_using_cumulative_reward_per_turn`
- `algorithm.adv_estimator=grpo`
- `algorithm.filter_groups.*`
- `trainer.rollout_data_dir`
- `trainer.default_local_dir`

## 3. main_ppo 和 RayPPOTrainer

入口文件：

```text
verl_agents/verl/trainer/main_ppo.py
```

执行流程：

1. Hydra 读取 `ppo_trainer.yaml` 和脚本覆盖参数。
2. `run_ppo(config)` 初始化 Ray。
3. 创建远程 `TaskRunner`。
4. `TaskRunner.run()`：
   - 下载或复制模型路径；
   - 初始化 tokenizer 和 processor；
   - 根据配置选择 FSDP 或 Megatron worker；
   - 创建 actor/rollout/ref/critic/reward worker；
   - 创建 reward manager；
   - 实例化 `RayPPOTrainer`；
   - 调 `trainer.fit()`。

主要训练类：

```text
verl_agents/verl/trainer/ppo/ray_trainer.py
```

这里负责 dataloader、rollout、reward、advantage、filter、actor update、保存 checkpoint、validation 和日志。

## 4. 数据读取与 prompt 构造

训练数据加载在：

```text
verl_agents/verl/utils/dataset/rl_dataset.py
verl_agents/verl/utils/dataset/rl_dataset_wo_mm_hint.py
```

`RayPPOTrainer._create_dataloader()` 根据：

```text
data.with_mm_hint
```

选择 dataset 类：

- `with_mm_hint=True`：使用 `RLHFDataset`
- `with_mm_hint=False`：使用 `RLHF_wo_mm_hint_Dataset`

### 4.1 带初始图模式

文件：

```text
verl_agents/verl/utils/dataset/rl_dataset.py
```

这个模式主要对应 PyVision-Image 当前较常用的训练方式。

当数据还不是 RL 格式时，`transfer_to_rl_form_image_w_mm_hint()` 会把原始 image QA 数据转换成 agent 训练格式。

核心字段：

```python
new_item["prompt"] = [{"content": prompt, "role": "user"}]
new_item["env_name"] = "pyvision_gym_w_image_hint"
new_item["reward_model"] = {"ground_truth": answer, "style": "model"}
new_item["extra_info"] = {
    "answer": answer,
    "index": int(item["id"]),
    "question": question,
    "split": "train",
}
new_item["mm_hint"] = {
    "hint_path": image_path,
    "hint_type": "image",
}
```

prompt 构造逻辑大致是：

```python
prompt = "<image>\n" + prompt_prefix.format(query=question, width=width, height=height)
```

其中 `<image>` 会在 `_build_messages_pyvision()` 中被替换成：

```text
<image_clue_0>
真实 image content
</image_clue_0>
```

这意味着模型初始就能看到原图。同时原图还会保存到 `origin_multi_modal_data`，之后注入 Python runtime 为 `image_clue_0`。

### 4.2 无初始图 / 视频模式

文件：

```text
verl_agents/verl/utils/dataset/rl_dataset_wo_mm_hint.py
```

这个类支持两类：

- image without init image：图片不直接喂给模型，只进入工具 runtime。
- video without init video：视频不直接喂给模型，只给视频元信息，完整视频进入工具 runtime。

图片转换函数：

```text
transfer_to_rl_form_image()
```

使用 prompt key：

```text
vis_tool_with_img_info_wo_init_image_v2
```

env：

```text
pyvision_gym_wo_image_hint
```

runtime 变量：

```text
image_hint_0
```

视频转换函数：

```text
transfer_to_rl_form_video()
```

使用 prompt key：

```text
vis_tool_with_img_info_video_v4
```

视频 prompt 里只写元信息：

```text
Frame Width
Frame Height
Video Length
Sample FPS
The original video has been read into the global variable `video_clue_0`.
```

env：

```text
pyvision_gym_wo_video_hint
```

runtime 变量：

```text
video_clue_0
```

这里的关键是：原始训练设置下，视频主线并不是初始把视频帧喂给 VLM，而是让模型通过 Python code 主动从 `decord.VideoReader` 中取帧，再用 `plt.show()` 把图像 observation 返回给自己。

## 5. prompt 模板

prompt 模板文件：

```text
verl_agents/verl/utils/dataset/rl_system_prompt_template.json
```

里面有多个版本，历史包袱比较多。和 PyVision 视觉工具训练最相关的是：

- `vistool_with_img_info_v2`：图像，初始给图，runtime 有 `image_clue_i`。
- `vistool_with_img_info_v3`：在 v2 基础上增强了 eval 时发现的问题约束，比如 `image_clue_i` 是 PIL 对象，不要 `Image.open("image_clue_0.png")`。
- `vis_tool_with_img_info_wo_init_image_v2`：图像，但初始不直接给图，只给 `image_hint_i`。
- `vis_tool_with_img_info_video_v4`：视频训练原始主线，初始不给视频帧，只给视频元信息和 `video_clue_j`。
- `vis_tool_with_img_info_video_v8`：在我们后续 eval 调试中形成的版本，语义变成初始也给 64 帧视频视觉输入，同时仍可用 Python inspect 全视频。

训练时模型工具调用格式不是 Qwen3-VL 官方 tools JSON，而是项目自己的 code block 格式：

```text
<code>
```python
...
```
</code>
```

最终答案格式：

```text
<answer>
\boxed{...}
</answer>
```

工具返回格式大致是：

```text
<tool_response>
<interpreter>
Text Result:
...
Image Result:
<image_clue_k><image></image_clue_k>
</interpreter>
</tool_response>
```

这也是我们后续 eval 代码要对齐训练 prompt，而不是只依赖 `<tool_call>` JSON 的原因。

## 6. agent rollout 接入点

普通 verl rollout 是一次性 generate 出完整 response。这个项目在 vLLM rollout 中加了 agent 分支。

文件：

```text
verl_agents/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py
```

核心判断：

```python
if self.config.agent.activate_agent:
    agent_proto = agent_rollout_loop(...)
else:
    outputs = self.inference_engine.generate(...)
```

当：

```text
actor_rollout_ref.rollout.agent.activate_agent=True
```

时，进入：

```text
verl_agents/verl/workers/agent/parallel_env.py
```

中的：

```python
agent_rollout_loop(...)
```

## 7. 多轮 agent rollout 逻辑

`agent_rollout_loop()` 是训练中最关键的部分之一。

它做的事可以概括为：

1. 复制 batch 中每个 prompt 的 `n` 条采样轨迹。
2. 为每条轨迹创建独立工具环境。
3. 每一轮调用 vLLM 生成 assistant action。
4. action 可能是 `<answer>`，也可能是 `<code>`。
5. 如果是 code，则执行工具。
6. 工具返回文本或图片 observation。
7. observation 拼回 vLLM prompt token 和训练侧 token 序列。
8. 继续下一轮，直到 answer、超长、超轮数、工具错误等。

agent rollout 的 stop 词来自配置：

```yaml
actor_rollout_ref:
  rollout:
    agent:
      custom_stop: ['</code>']
```

代码里设置：

```python
agent_sampling_params.include_stop_str_in_output = True
agent_sampling_params.skip_special_tokens = False
agent_sampling_params.spaces_between_special_tokens = False
agent_sampling_params.n = 1
agent_sampling_params.max_tokens = min(single_response_max_tokens, response_length)
```

外层 GRPO 的 `rollout.n` 仍然表示一个 prompt 采样多少条轨迹，但 agent 循环内部每次交互只生成一段 action。

### 7.1 action token 和 observation token 的区别

训练时需要区分：

- 模型自己生成的 action token；
- 工具返回的 observation token。

因此 `agent_rollout_loop()` 维护：

- `running_states`
- `running_action_masks`
- `running_attn_masks`
- `reward_tensor_list`

模型输出 action 时：

```text
action_mask = 1
```

工具 observation 拼回上下文时：

```text
action_mask = 0
attention_mask = 1
```

这表示 observation 可以被模型看见，但不参与 policy loss。

## 8. 工具注册与工具环境

工具注册入口：

```text
verl_agents/verl/workers/agent/__init__.py
```

这里 import 一堆 env 的目的主要是触发 `ToolMeta` 元类注册。

工具基类：

```text
verl_agents/verl/workers/agent/tool_envs.py
```

`ToolBase.registry` 是全局注册表。数据样本里有：

```text
env_name
```

rollout 时 `ParallelEnv.reset()` 根据：

```python
ToolBase.create(tool_name)
```

创建对应工具。

PyVision 当前主线常用工具：

```text
pyvision_gym_w_image_hint
pyvision_gym_wo_image_hint
pyvision_gym_wo_video_hint
```

对应文件：

```text
verl_agents/verl/workers/agent/envs/agents_x/safe_persis_python_exe_tool_w_image_hint.py
verl_agents/verl/workers/agent/envs/agents_x/safe_persis_python_exe_tool_wo_image_hint.py
verl_agents/verl/workers/agent/envs/agents_x/safe_persis_python_exe_tool_wo_video_hint.py
```

## 9. Python 工具 runtime

这三个 PyVision 工具都采用“持久化子进程 + Python exec”的方式。

核心结构：

- `PersistentWorker`
  - 每条轨迹一个 worker process；
  - 通过 multiprocessing queue 接收代码；
  - 保留 runtime 状态；
  - 支持跨轮变量复用。
- `SafeImageRuntime`
  - 初始化 Python 全局变量；
  - 注入图片或视频；
  - 替换 `plt.show()` 为内部捕获函数；
  - 捕获 stdout 和 matplotlib figure。

虽然类名叫 `SafeImageRuntime`，它并不是完整安全沙箱。它主要做：

- 子进程隔离；
- 禁止明显危险调用，比如 `input()`、`os.system()`、`subprocess()`；
- 捕获 `plt.show()` 生成的图片；
- 超时返回 error。

### 9.1 图像带初始图工具

文件：

```text
safe_persis_python_exe_tool_w_image_hint.py
```

工具名：

```text
pyvision_gym_w_image_hint
```

模型初始看到图片。runtime 也注入：

```text
image_clue_0
```

当模型调用：

```python
plt.show()
```

工具会把当前 matplotlib figure 转成图片 observation，并追加成新的：

```text
image_clue_1
image_clue_2
...
```

### 9.2 图像无初始图工具

文件：

```text
safe_persis_python_exe_tool_wo_image_hint.py
```

工具名：

```text
pyvision_gym_wo_image_hint
```

模型初始不直接看到原图。runtime 注入：

```text
image_hint_0
```

模型必须用 code 主动展示整图、局部 crop 或处理后图像。

### 9.3 视频工具

文件：

```text
safe_persis_python_exe_tool_wo_video_hint.py
```

工具名：

```text
pyvision_gym_wo_video_hint
```

runtime 注入：

```python
video_clue_0 = decord.VideoReader(video_path, ctx=cpu(0))
```

模型可以写：

```python
num_frames = len(video_clue_0)
fps = video_clue_0.get_avg_fps()
frames = video_clue_0.get_batch(indices).asnumpy()
plt.imshow(frames[0])
plt.show()
```

注意 decord 的 API 是 `video_clue_0[idx]` 或 `get_batch(indices)`，不是 `get_frame()`。这是我们 eval 时遇到过的一个模型代码错误。

## 10. 工具输出如何拼回模型上下文

工具执行后返回的结果有三种形态：

1. 字符串：纯文本 observation。
2. OpenAI/Qwen chat list：结构化文本消息。
3. dict：包含 `chat` 和 `multi_modal_data`，用于图片 observation。

处理函数：

```text
verl_agents/verl/workers/agent/parallel_env.py
execute_tool_call()
```

对于图片 observation，工具返回类似：

```python
{
    "prompt": "",
    "chat": obs_chat,
    "multi_modal_data": {"image": [PIL.Image, ...]},
}
```

随后 `_preprocess_multi_modal_inputs()` 会：

1. 把 `<image>` 替换成 Qwen-VL 的视觉占位：

```text
<|vision_start|><|image_pad|><|vision_end|>
```

2. 用 processor 把 PIL 图片转成：

```text
pixel_values
image_grid_thw
```

3. 分别维护：

- vLLM 下一轮 generate 需要的 `prompt_token_ids_vllm`；
- 训练侧 forward/logprob 需要的 `prompt_token_ids_model` 和 `multi_modal_inputs`。

这也是训练代码复杂的地方：vLLM generate、模型训练 forward、Qwen-VL mRoPE 位置编码都需要一致。

## 11. mRoPE 和多模态一致性

Qwen2.5-VL 需要视觉位置编码。训练中如果工具返回图片，必须保证：

- 文本中有正确数量的视觉占位符；
- `image_grid_thw` 中有对应数量的图像张量。

检查函数：

```text
check_vision_tokens_num_images_num_consistency()
```

如果一致，则用：

```text
verl.models.transformers.qwen2_vl.get_rope_index()
```

重新计算视觉 position ids。

如果不一致，则生成 dummy position ids，并记录：

```text
is_vision_token_nums_image_nums_consistent=False
```

这个字段后面可以作为 filter metric 筛掉坏轨迹。

## 12. 结束原因与轨迹状态

`agent_rollout_loop()` 用：

```text
EndReasonEnum
```

记录轨迹为什么结束：

```text
ON_GONIG
DONE
OVER_LENGTH
EXCEED_MAX_TURNS
EXCEED_MAX_IMAGE_NUM_32
ERROR_IN_ACTION
```

这些状态进入 `DataProto.non_tensor_batch["end_reason"]`，后续用于：

- 日志统计；
- filter groups；
- 判断哪些 rollout 质量较差。

## 13. 工具奖励

工具本身每成功执行一次，可以返回一个过程奖励：

```text
tool_using_cumulative_reward_per_turn
```

它在工具成功执行 code 后返回：

```python
return obs, self.tool_using_cumulative_reward_per_turn, False, {"status": "success"}
```

agent rollout 会把这个 reward 写入：

```text
env_reward
```

但 reward manager 里有关键设计：工具奖励不是无条件加入。

文件：

```text
verl_agents/verl/workers/reward_manager/naive.py
```

逻辑是：

```python
if score["is_answer_right"]:
    reward_tensor[i] += env_reward_tensor[i]
```

也就是说：

- 答案正确：准确性 reward + 累积工具 reward。
- 答案错误：不给工具过程奖励。

这和 README 里“accumulative tool reward 防止 interaction collapse”的设计一致，同时避免模型学会无意义地多调用工具。

## 14. 最终答案 reward

默认 reward 分发在：

```text
verl_agents/verl/utils/reward_score/__init__.py
```

根据 `data_source` 选择不同 scorer。多模态视觉相关数据通常走：

```text
multi_task_verifier.compute_score(...)
```

相关 data_source 包括：

```text
vstar
vl_agent
chart
zebra_cot
vigorl
deepeyes
math_8k_verified
barc
wemath-standard
wemath-pro
minio3
vsi
longvila
ecd
v-interaction
mathvision
hrbench-4k
hrbench-8k
```

`multi_task_verifier.py` 和 `general_verifier.py` 都会从模型输出中抽取：

```text
<answer>
\boxed{...}
</answer>
```

然后用 rule、math_verify 或 LLM-as-a-judge 判断是否正确。

LLM-as-a-judge 的服务配置来自环境变量：

```text
LLM_AS_A_JUDGE_CONFIG_PATH
```

README 和训练脚本都强调训练前要先启动 judge server。

## 15. GRPO 与 oversampling/filtering/ranking

训练脚本一般设置：

```text
algorithm.adv_estimator=grpo
actor_rollout_ref.rollout.n=8 或 16
```

这表示每个 prompt 采样多条轨迹，用组内 reward 差异估计 advantage。

为了避免所有样本都无差异，项目加入了 oversampling/filtering/ranking 逻辑。

相关文件：

```text
verl_agents/verl/trainer/ppo/ray_trainer.py
verl_agents/verl/trainer/ppo/filter_fn_utils.py
verl_agents/verl/trainer/ppo/metric_utils_oversample_pool.py
```

训练脚本里常见配置：

```bash
enable_filter_groups=True
filter_groups_metric='seq_reward,hasimage,trajlength,vtoken_images_num_consis,end_reason'
end_reason_filter_reserve_names='DONE,EXCEED_MAX_TURNS,ERROR_IN_ACTION'
max_num_gen_batches=0
std_sort_enable=True
```

我对这部分的理解是：

1. 先生成一个 rollout batch。
2. 计算 reward 和 advantage。
3. 如果开启 filter groups，则按配置指标过滤轨迹。
4. 如果过滤后数量不足，就继续生成更多 batch，直到凑够目标训练轨迹数。
5. 如果生成过多，可以根据 `sample_level_stds` 排序，优先保留组内 reward 方差更大的样本。

重要 filter：

- `seq_reward`：根据序列 reward 过滤或产生统计。
- `hasimage`：是否真的产生过图片 observation。
- `trajlength`：轨迹长度是否超限。
- `vtoken_images_num_consis`：视觉 token 和图片数量是否一致。
- `end_reason`：根据 DONE、OVER_LENGTH、ERROR 等结束原因筛。

这部分是 PyVision-RL 防止 interaction collapse 的关键工程设计：不要让训练全被“不调用工具”或“无效工具调用”的轨迹占满。

## 16. actor loss 与 action_mask

actor 更新在：

```text
verl_agents/verl/workers/actor/dp_actor.py
```

普通 PPO loss 只应该作用在模型生成的 token 上，而不应该作用在工具 observation token 上。

因此 agent rollout 返回：

```text
action_mask
```

actor 里使用：

```python
action_or_attn_mask = data["action_mask"] if "action_mask" in data else data["attention_mask"]
response_mask = action_or_attn_mask[:, -response_length:]
```

然后用 `response_mask` 计算 policy loss。

这样：

- 模型 action token 参与 loss；
- 工具 observation token 只作为上下文，不参与 loss。

代码里还有 `interaction_budget` 和 `overbudget_masking` 的痕迹，但当前看到的实现中相关 masking 逻辑大多被注释掉了，因此实际是否生效要看后续是否恢复那段代码。训练脚本里仍然会传：

```text
actor_rollout_ref.actor.interaction_budget
actor_rollout_ref.actor.overbudget_masking
```

但就当前 `dp_actor.py` 来看，它们更像保留参数。

## 17. 训练数据形态

README 给的 validation 格式是原始 JSON，例如 image：

```json
{
  "id": "0",
  "question": "...",
  "answer": "A",
  "ability": "visual_search",
  "data_source": "vstar",
  "image_path": "/path/to/image.jpg"
}
```

video：

```json
{
  "id": "0",
  "question": "...",
  "answer": "A",
  "ability": "spatial_reasoning",
  "data_source": "vsi",
  "video_path": "/path/to/video.mp4"
}
```

dataset 类会把它们转成 RL agent 格式，核心包含：

```text
prompt
data_source
ability
env_name
reward_model
extra_info
mm_hint
```

如果训练数据本身已经带 `mm_hint`，转换函数会直接返回，不重复处理。

训练脚本中有不少 parquet 路径，但当前 dataset 代码读取逻辑主要是 `json.load()`。这说明仓库当前代码和历史脚本之间有一定演进痕迹：README 新格式偏 JSON，部分旧实验脚本仍保留 parquet 路径变量。实际运行哪个版本，需要以当前脚本和数据真实格式为准。

## 18. 图像训练和视频训练的关键差异

图像训练主要有两种：

1. 初始给图：
   - prompt 中含 `<image>`；
   - 模型一开始能看原图；
   - runtime 也注入 `image_clue_0`；
   - env 为 `pyvision_gym_w_image_hint`。

2. 初始不给图：
   - prompt 只给图像元信息；
   - runtime 注入 `image_hint_0`；
   - 模型必须调用工具看图；
   - env 为 `pyvision_gym_wo_image_hint`。

视频训练原始主线更接近第二种：

- 初始不喂 video frames；
- prompt 只给 `Frame Width/Height/Video Length/FPS`；
- runtime 注入 `video_clue_0 = decord.VideoReader(video_path)`；
- 模型通过代码抽帧并 `plt.show()`；
- env 为 `pyvision_gym_wo_video_hint`。

这也是我们之前做 eval 时讨论“是否初始给 64 帧视频”的核心区别：原训练代码中的 v4 视频 prompt 更偏按需取帧；我们 eval 后来的 v8 是为了 Qwen3-VL online eval 做的适配，不一定等价于最早训练设置。

## 19. rollout 保存与日志

训练脚本常设置：

```text
trainer.rollout_data_dir
trainer.the_first_batch_rollout_data_dir
trainer.the_oversample_data_pool_rollout_data_dir
```

`ray_trainer.py` 会在配置存在时 dump batch generations。这些 rollout 日志对于调试很重要，因为可以看到：

- prompt；
- response；
- tool call；
- tool observation；
- final answer；
- reward；
- end reason；
- tool count。

日志指标还包括：

- `agent/tool_call_mean`
- `agent/tool_call_max`
- `agent/tool_call_min`
- data_source 级别 tool call；
- end_reason 分布；
- oversample pool 指标；
- reward/advantage/entropy/throughput。

## 20. checkpoint 与合并

训练保存由 `trainer.default_local_dir` 和 `actor_rollout_ref.actor.checkpoint.contents` 控制。

脚本里常见：

```text
actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra']
```

这表示 checkpoint 里除了训练状态，也保存 HF 模型。仓库还有：

```text
verl_agents/scripts/model_merger.py
verl_agents/scripts/run_merge.sh
```

用于把训练 checkpoint 合并为可推理的 HF 权重。

## 21. 当前 eval 代码和训练代码的关系

我们后续新增的 eval 脚本并不是训练链路的一部分，但设计上要对齐训练。

关键对齐点：

- 使用 `<code>```python ...```</code>`，而不是只使用 `<tool_call>` JSON。
- 使用 `<tool_response><interpreter>...</interpreter></tool_response>`。
- 图片工具返回后，给模型真实图片 observation，而不是纯文本 base64。
- 图像初始输入用 `<image_clue_0>` 包裹。
- runtime 注入 `image_clue_0` 或 `video_clue_0`。
- 视频要明确 decord 的正确取帧方式。

当前 eval 中为 Qwen3-VL 做了一些额外工程修补，例如：

- `_rewrite_virtual_clue_opens`
- `_rewrite_pil_indexing`
- vLLM online 并发；
- resume / rerun / merge；
- baseline 和 summary。

这些是评测侧增强，不属于原始训练算法本体。

## 22. 我对训练主链路的一句话总结

这个项目的训练链路可以概括为：

```text
原始图像/视频 QA 数据
  -> dataset 转成带 env_name/mm_hint/reward_model 的 agent prompt
  -> vLLM 生成一段 action
  -> Python 工具执行 <code>
  -> 文本/图片 observation 拼回上下文
  -> 多轮循环直到 <answer> 或异常结束
  -> LLM-as-a-judge / verifier 给最终答案 reward
  -> 答案正确时叠加工具过程奖励
  -> GRPO 计算 advantage
  -> filter/ranking 选择有效轨迹
  -> FSDP actor 更新
```

其中最关键的工程点有三个：

1. **多轮上下文维护**：action token 和 observation token 要分开 mask，图片 observation 还要同步维护 vLLM 输入和训练侧 `multi_modal_inputs`。
2. **工具奖励设计**：工具奖励只在答案正确时生效，避免 reward hacking。
3. **轨迹过滤**：通过 hasimage、trajlength、end_reason、视觉 token 一致性、reward 方差等指标，尽量保留对 agent 行为有训练价值的轨迹。

## 23. 仍需注意的不确定点

我目前的理解来自当前仓库代码和脚本，仍有几个点需要在真实训练日志中确认：

1. 部分历史训练脚本还保留 parquet 路径，但当前 dataset 代码看起来主要按 JSON 读取。实际训练数据格式可能取决于当时分支或运行环境。
2. `interaction_budget` / `overbudget_masking` 在脚本里传入，但当前 actor loss 中相关逻辑基本被注释，实际未必生效。
3. `vis_tool_with_img_info_video_v8` 是我们后续 eval 适配产生的 prompt，原始视频训练更像使用 v4 的“不初始给视频帧”方案。
4. `LLM_AS_A_JUDGE_BASE` 和 `LLM_AS_A_JUDGE_CONFIG_PATH` 都出现过，真正 reward manager 使用的是 config json 中的 `base_url/model_name`。
5. 工具 runtime 是基础隔离，不是严格安全沙箱；训练时假设模型和数据可信。

