# Qwen3-VL Agent Eval 工具代码错误分析

本文档记录当前 vLLM online agent eval 中 Python 工具调用错误的类型、已采取的修复、仍然存在的问题，以及后续重跑/合并建议。

## 当前统计

统计文件：

```text
eval/results/qwen3vl_vllm_online_agent/qwen3vl_vllm_online_agent_no_lvb.jsonl
```

当前共 600 个 case，六个数据集各 100 个：

```text
总数: 600
正确数: 387
准确率: 64.50%
success: 552
tool_execution_error: 28
no_code_or_answer: 20
```

按数据集统计：

| 数据集 | success | tool_execution_error | no_code_or_answer |
| --- | ---: | ---: | ---: |
| vstar_bench | 91 | 7 | 2 |
| VideoMME | 95 | 3 | 2 |
| HR-Bench-4K | 94 | 6 | 0 |
| HR-Bench-8K | 95 | 4 | 1 |
| MathVista | 93 | 1 | 6 |
| MathVision | 84 | 7 | 9 |

当前剩余工具错误类型：

| 类型 | 数量 |
| --- | ---: |
| 引用不存在的 clue 变量或未定义变量 | 10 |
| 其他工具代码错误 | 9 |
| 虚拟图片名被当作文件读取 | 3 |
| 导入环境中不存在的可选库 | 3 |
| 图像/视频索引越界 | 2 |
| numpy shape/pixel 逻辑错误 | 1 |

## 错误类型与处理方式

### 1. 虚拟图片名被当作文件读取

典型错误：

```text
FileNotFoundError: No such file or directory: 'image_clue_0.png'
FileNotFoundError: No such file or directory: 'image1.png'
```

典型错误代码：

```python
image = Image.open("image_clue_0.png")
image = Image.open("image_clue_0")
image = Image.open("image1.png")
```

原因：

`image_clue_0` 在 runtime 中是已经加载好的 `PIL.Image.Image` 对象，不是磁盘路径。模型有时会把变量名误当成文件名。

当前解决：

已在 runtime 中扩展 `_rewrite_virtual_clue_opens`，覆盖：

```python
Image.open("image_clue_0")
Image.open("image_clue_0.png")
Image.open("image_clue_0.jpg")
Image.open("image1.png")
imread("image_clue_0")
```

会自动改写为：

```python
image_clue_0.copy()
np.array(image_clue_0.copy())
```

同时新增 prompt `vistool_with_img_info_v3`，明确提示 `image_clue_i` 是 PIL 对象，不要重新打开虚拟文件名。

剩余风险：

如果模型生成更复杂的虚拟文件名，例如 `./image_clue_0.png`、`/tmp/image1.png`、`image_1.png`，当前规则不一定覆盖。若这种错误继续出现，可以继续扩展 rewrite pattern。

### 2. PIL.Image 被当作 numpy 数组切片或比较

典型错误：

```text
TypeError: 'Image' object is not subscriptable
TypeError: '>' not supported between instances of 'Image' and 'int'
```

典型错误代码：

```python
image_clue_0[:, :, 0]
image_rgb[:, :, 0]
image_rgb > 200
```

原因：

`image_clue_0` 和 `image_rgb = image.convert("RGB")` 都是 PIL Image。像素级切片和阈值比较需要先转成 numpy array。

当前解决：

已新增 `_rewrite_pil_indexing`，自动处理明显模式：

```python
image_clue_0[:, :, 0] -> np.array(image_clue_0)[:, :, 0]
image_rgb[:, :, 0]   -> np.array(image_rgb)[:, :, 0]
image_rgb > 200      -> np.array(image_rgb) > 200
```

它也会追踪简单赋值链：

```python
image = image_clue_0
image_rgb = image.convert("RGB")
```

prompt `v3/v8` 也提示：切片、mask、channel access、numeric comparison 前先 `np.array(image_clue_0)`。

剩余风险：

复杂表达式、函数返回值、列表/字典中保存的 PIL Image 不一定能自动追踪。此类属于“可继续扩展 sanitizer，但不能完全保证”的问题。

### 3. 导入环境中不存在或不建议使用的库

典型错误：

```text
ModuleNotFoundError: No module named 'skimage'
ModuleNotFoundError: No module named 'pytesseract'
ImportError: cv2 binding error
```

典型错误代码：

```python
from skimage.filters import threshold_otsu
import pytesseract
import cv2
```

原因：

模型倾向调用 OCR/CV 工具包，但 eval 环境不一定安装这些包；其中 `pytesseract` 即使安装 Python 包，也通常还需要系统级 tesseract binary。

当前解决：

prompt `v3/v8` 已明确写：

```text
Do not import OCR/CV or optional packages such as cv2, skimage, or pytesseract.
```

建议：

暂时不建议通过安装包解决。原因是：

1. `pytesseract` 依赖系统 binary，环境不稳定。
2. `skimage/cv2` 会让模型走更复杂的 CV 代码，容易带来更多 shape/index 错误。
3. 为了和训练/评测设置一致，工具能力最好限制在 `numpy/matplotlib/PIL object/torch/decord`。

可解决性：

部分可解决。可以在 runtime 层禁止 import 后返回更友好的错误，或者在执行前把 `skimage.threshold_otsu` 这类常见用法替换成 numpy 简易实现。但这会逐步变成一个复杂兼容层，短期优先级低于减少工具调用错误和提升 prompt 约束。

### 4. 引用不存在的 clue 变量或未定义变量

典型错误：

```text
NameError: name 'image_clue_1' is not defined
NameError: name 'answer' is not defined
NameError: name 'bus_x' is not defined
NameError: name 'bag_color' is not defined
NameError: name 'green' is not defined
NameError: name 'cv2' is not defined
```

原因：

这类错误分两种：

1. 模型误以为有多个图像 clue，例如 `image_clue_1`，但当前单图任务只有 `image_clue_0`。
2. 模型写了不完整代码，例如前面没有定义 `answer/bus_x/bag_color`，后面直接使用。

当前解决：

`image_clue_1` 类问题可以在 prompt 中进一步强调单图任务只有 `image_clue_0`。但未定义局部变量通常来自模型代码逻辑不完整，不能通过简单 rewrite 稳定修复。

可解决性：

部分可解决：

- `image_clue_1` 可以考虑在单图任务中自动 alias 到 `image_clue_0`，但这可能掩盖多图任务真实错误，不建议默认做。
- `answer/bus_x/bag_color` 这类未定义变量暂时不建议自动修复，因为系统无法知道模型想表达什么。

建议处理：

依靠 rerun。多数这类错误是采样随机性导致的代码不完整，重新生成可能会消失。

### 5. 图像/视频索引越界

典型错误：

```text
IndexError: Out of bound indices: [2875]
IndexError: image index out of range
IndexError: index 18 is out of bounds for axis 0 with size 18
```

典型错误代码：

```python
indices = np.linspace(start_frame, end_frame, 10, dtype=int)
frames = video_clue_0.get_batch(indices).asnumpy()
```

原因：

模型根据时间或假设长度计算 frame index，但没有 clamp 到 `[0, len(video)-1]`。

当前解决：

video prompt 已提示用：

```python
num_frames = len(video_clue_0)
indices = np.linspace(0, num_frames - 1, ...)
```

但模型仍可能在细看某段时越界。

可解决性：

可以进一步在 runtime 中做轻量 helper，例如提供：

```python
safe_get_batch(video, indices)
```

或者 rewrite `video.get_batch(indices)` 前自动 clamp `indices`。但自动改写任意变量名比较冒险，可能影响模型预期。当前建议先通过 prompt + rerun 处理。

### 6. numpy shape / pixel 逻辑错误

典型错误：

```text
ValueError: operands could not be broadcast together
ValueError: too many values to unpack
AxisError: axis 2 is out of bounds for array of dimension 2
```

原因：

模型写的图像处理逻辑本身不严谨，例如：

- 把灰度图当 RGB 图；
- mask 维度和图像维度不一致；
- 对裁剪/聚合后的二维数组继续访问 RGB channel。

当前解决：

`_rewrite_pil_indexing` 能解决“忘记转 numpy”的一部分，但不能保证模型的图像算法正确。

可解决性：

暂时不能稳定自动解决。因为这类问题涉及模型具体算法意图，自动修改可能产生更隐蔽的错误答案。建议通过 rerun 或降低模型写复杂 CV 代码的倾向来处理。

### 7. 其他工具代码错误

当前剩余的“其他”包括：

```text
IndexError: too many indices for array
IndexError: index 550 is out of bounds
TypeError: labeled_comprehension() got an unexpected keyword argument 'dtype'
ValueError: Calling nonzero on 0d arrays is not allowed
```

原因：

这些通常是模型生成了过度复杂、未经验证的程序化视觉分析代码。它们并不是 runtime 协议问题，而是模型代码质量问题。

可解决性：

短期不建议逐个规则修复。更好的方向是：

1. prompt 约束模型少写复杂 CV 代码；
2. 鼓励模型优先用 `plt.imshow/crop/grid` 进行视觉观察；
3. 对失败 case rerun；
4. 必要时让工具错误返回 observation，而不是终止整条样本。

## 已经采取的修复

当前代码层修复：

1. 扩展 `_rewrite_virtual_clue_opens`：
   - 支持 `image_clue_0.png/.jpg`；
   - 支持 `image1.png`；
   - 支持 `imread("image_clue_0")`。

2. 新增 `_rewrite_pil_indexing`：
   - 自动把明显 PIL 切片/比较包装成 `np.array(...)`；
   - 支持简单变量链追踪。

3. 新增 prompt：
   - `vistool_with_img_info_v3`
   - `vis_tool_with_img_info_video_v8`

4. online eval 切换使用上述 prompt key，并显式启用 runtime rewrite。

## 仍然不能稳定解决的问题

暂时不能可靠自动解决：

1. 未定义局部变量，例如 `answer/bus_x/bag_color`。
2. 模型自行设计的复杂 CV 算法 bug。
3. 灰度/RGB 维度判断错误。
4. `skimage/pytesseract/cv2` 等依赖型代码。
5. 视频精确时间段采样时的越界和语义错误。

这些问题的共同特点是：它们不是简单协议不匹配，而是模型生成代码的程序逻辑错误。自动修复可能会改变语义，导致“代码不报错但答案更错”。

## 建议的 rerun 流程

当前建议按轮次重跑，不要在同一个 split 输出目录里无限 `--resume`。

第一轮/当前 split 输出目录：

```text
eval/results/qwen3vl_vllm_online_agent/rerun_no_lvb_splits/
```

如果某个 split 中途断掉，例如 HRBench，可以只补跑它：

```bash
bash eval/run_qwen3vl_vllm_online_agent_rerun_splits.sh \
  --splits hrbench \
  --resume
```

四个 split 都完成后，合并回主文件：

```bash
python eval/merge_qwen3vl_rerun_results.py --in-place
```

合并后检查还剩多少需要重跑：

```bash
python eval/qwen3vl_vllm_online_agent_eval.py \
  --dry-run-data \
  --dry-run-preview-limit 0 \
  --rerun-from-results eval/results/qwen3vl_vllm_online_agent/qwen3vl_vllm_online_agent_no_lvb.jsonl \
  --rerun-mode both
```

如果还要第二轮，推荐换新输出目录：

```bash
OUTPUT_DIR=eval/results/qwen3vl_vllm_online_agent/rerun_no_lvb_splits_round2 \
bash eval/run_qwen3vl_vllm_online_agent_rerun_splits.sh
```

然后按 round2 合并：

```bash
python eval/merge_qwen3vl_rerun_results.py \
  --rerun-glob 'eval/results/qwen3vl_vllm_online_agent/rerun_no_lvb_splits_round2/ours_rerun_*.jsonl' \
  --in-place
```

原因：

如果同一个目录里继续 `--resume`，已经 rerun 过但仍然错误的 case 会被跳过。换新目录可以确保每轮都针对当前主文件中的剩余错误重新生成。

## 后续可考虑的改进

1. 工具执行错误不立即终止样本，而是把错误文本作为 `<tool_response>` 返回给模型，让模型自我修正代码。
2. 降低工具代码复杂度：prompt 中强调优先 `imshow/crop/grid`，少写颜色阈值/连通域/复杂 CV。
3. 提供少量稳定 helper：
   - `to_array(image_clue_0)`
   - `show_crop(image_clue_0, box)`
   - `safe_video_batch(video_clue_0, indices)`
4. 对工具错误 case 使用更低 temperature 或单独 retry 策略。
5. 单独统计“工具错误但 prediction 已正确”的 case，避免为了消除 status 反复重跑已经答对的样本。
