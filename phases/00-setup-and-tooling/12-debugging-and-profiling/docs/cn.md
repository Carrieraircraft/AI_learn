下面把这一课「掰开揉碎」讲。既讲**它为什么这么设计**，也讲**每段代码到底在干嘛、你以后会怎么用**。你有嵌入式 + 全栈底子，我会把它和你熟悉的东西对照。

---

## 一、这一课的核心世界观

一句话：**AI 的 bug 大多不报错。**

你写嵌入式/后端时，bug 通常表现为：崩溃、段错误、异常、栈回溯——系统主动告诉你「这里错了」。

AI 训练不是这样。文档开头那句是灵魂：

> The worst AI bugs don't crash. They train silently on garbage and report a beautiful loss curve.

典型翻车现场：训练跑了 8 小时，没有任何报错，loss 曲线还挺漂亮，结果模型学出来只会输出「所有输入的平均值」。原因可能只是：
- 一个张量忘了搬到 GPU（`.cuda()`）
- 忘了 `.detach()`，把计算图拖着不放
- 标签（label）不小心混进了特征（feature）——数据泄漏

这些**都不会抛异常**。所以这课的目的不是教你「怎么读栈回溯」，而是教你**主动去探测这些沉默的错误**。

---

## 二、三层调试模型（The Concept）

文档给了一个金字塔（我用文字画）：

```text
第 3 层：训练动态   —— loss 曲线、梯度范数、激活分布   （TensorBoard）
第 2 层：张量操作   —— shape / dtype / device / NaN、Inf （print、断言、hook）
第 1 层：普通 Python —— 断点、日志、性能分析、内存      （pdb、logging、cProfile）
```

关键洞察：

> 大多数人一上来就盯第 3 层（看 TensorBoard 曲线），但 **80% 的 bug 在第 1、2 层**。

对你的启发：别神化「AI 调试」。底座还是普通 Python 调试（你已会），只是**多了一层张量语义**（shape/dtype/device/NaN）要盯。这课就是在你已有的调试能力上，加装「张量探针」。

---

## 三、逐 Part 拆解

### Part 1：Print 调试（别看不起它）

```python
def debug_print(name, tensor):
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, "
          f"device={tensor.device}, "
          f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, "
          f"mean={tensor.mean().item():.4f}, "
          f"has_nan={tensor.isnan().any().item()}")
```

**为什么张量场景下 print 反而比断点好用？**
因为你一次要看的是**一组属性**：形状、数据类型、在哪个设备、数值范围、有没有 NaN。断点里逐个敲 `p x.shape`、`p x.dtype`… 很啰嗦；一行 `debug_print` 全打出来。

逐字段含义：
- `shape`：维度对不对（最高频 bug）
- `dtype`：`float32` / `float16` / `int64`，类型错会静默出问题
- `device`：`cpu` 还是 `cuda:0`
- `min/max/mean`：数值有没有爆炸/全 0/异常范围
- `has_nan`：有没有 NaN（沉默杀手）

`.item()` 是把「单元素张量」转成 Python 标量再打印。类比嵌入式里你插的 `printf("%d %d\n", a, b)` 探针——同样是「打点观测」，只是观测对象是张量。

---

### Part 2：条件断点（`breakpoint()` + pdb）

```python
if loss.item() > 100 or torch.isnan(loss):
    breakpoint()
```

**精髓在「条件」两个字。** 训练要跑 10000 步，你不可能每步都停。所以只在「异常发生的那一刻」才掉进调试器：loss 爆了或变成 NaN 才停。

`breakpoint()` 是 Python 3.7+ 内置，等价于 `import pdb; pdb.set_trace()`。掉进去后常用命令：

| 命令 | 作用 |
|------|------|
| `p outputs.shape` | 看形状 |
| `p loss.item()` | 看 loss 值 |
| `p torch.isnan(outputs).sum()` | 数有几个 NaN |
| `p model.fc1.weight.grad` | 看某层梯度 |
| `c` | 继续 |
| `q` | 退出 |

对你来说 pdb 不新鲜，新鲜的是「**条件触发**」这个用法——相当于硬件调试里的**条件断点/watchpoint**（某个寄存器满足条件才停），只不过条件是「loss 异常」。

---

### Part 3：日志（logging）

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)
```

为什么训练要用 logging 而不是 print？文档点得很实在：

> 训练凌晨 3 点挂了，你要的是一个**日志文件**，而不是早已滚出屏幕的终端输出。

两个 handler 的意思：`FileHandler` 写文件（事后追查），`StreamHandler` 打屏幕（实时看）。带时间戳、分级别（INFO/WARNING/ERROR）。这块你后端肯定熟，AI 里唯一区别是：**长时间无人值守**，所以文件持久化比平时更重要。

---

### Part 4：给代码段计时（Timer 上下文管理器）

```python
class Timer:
    def __enter__(self):
        self.start = time.perf_counter(); return self
    def __exit__(self, *a):
        print(f"[{self.name}] {time.perf_counter()-self.start:.4f}s")

with Timer("data loading"):
    batch = next(dataloader_iter)
```

用 `with` 包一段，自动打印耗时。`time.perf_counter()` 是高精度单调时钟（适合测时长）。

**这里有一条非常值钱的经验**，务必记住：

> 常见结论：数据加载占了 60% 的训练时间。解法是 DataLoader 设 `num_workers > 0`，**而不是换更快的 GPU**。

很多人 GPU 利用率上不去，第一反应是「卡不行」，其实是**数据喂不上**（CPU 端预处理/读盘是瓶颈）。这和你做系统优化时「先测量再优化，别猜」的原则完全一致。

---

### Part 5：cProfile 和 line_profiler

手动 Timer 只能测你「想到要测」的段。想全局找热点：

```bash
python -m cProfile -s cumtime train.py   # 按累计耗时排序所有函数调用
```

想精确到「每一行」：

```python
@profile                      # line_profiler 提供的装饰器
def train_step(model, data, target):
    ...
# 运行：kernprof -l -v train.py
```

- `cProfile`：函数级，标准库自带，先用它看「哪个函数慢」
- `line_profiler`：行级，需 `pip install`，锁定慢函数后再看「哪一行慢」

这套「先函数级、再行级」的下钻思路，和你平时用 perf/gprof 找热点一模一样。

---

### Part 6：内存分析（三种，分场景）

AI 有两种内存要管：**CPU 内存**和**GPU 显存**，工具不同。

**① CPU 内存 — `tracemalloc`（标准库）**
```python
tracemalloc.start()
...  # 你的代码
snapshot = tracemalloc.take_snapshot()
for stat in snapshot.statistics("lineno")[:10]:
    print(stat)   # 按行统计谁分配得最多
```

**② CPU 内存 — `memory_profiler`（逐行）**
```python
@profile
def load_data():
    raw = read_csv("data.csv")   # 看内存在哪一行猛涨
```

**③ GPU 显存 — PyTorch 内置**
```python
print(torch.cuda.memory_summary())
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")  # 实际用了多少
print(f"Cached:    {torch.cuda.memory_reserved()/1e9:.2f} GB")  # 预留了多少
```

注意 `allocated`（真正在用）vs `reserved`（PyTorch 向驱动预留的缓存池）的区别——PyTorch 会缓存显存不立刻还给系统，所以 `reserved ≥ allocated`。

**OOM（显存爆了）的处理顺序**，背下来能省很多命：
1. **先减 batch size**（永远第一招）
2. `torch.cuda.empty_cache()` 释放缓存
3. `del 大张量` 再 `empty_cache()`
4. 混合精度 `torch.cuda.amp`（显存砍一半）
5. gradient checkpointing（极深模型，用时间换显存）

你之前算过 4050 才 6GB，OOM 会是你的家常便饭，这一段对你尤其实用。

---

### Part 7：四大经典 AI Bug（本课最有价值的一节）

这是与「普通程序调试」差别最大的地方，建议重点吃透。

#### ① Shape Mismatch（形状不匹配）——最高频

```python
def check_shapes(model, sample_input):
    def make_hook(name):
        def hook(module, inp, out):
            in_shape  = inp[0].shape if isinstance(inp, tuple) else inp.shape
            out_shape = out.shape if hasattr(out, "shape") else type(out)
            print(f"  {name}: {in_shape} -> {out_shape}")
        return hook
    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(make_hook(name)))
    with torch.no_grad():
        model(sample_input)
    for h in hooks: h.remove()
```

核心机制是 **forward hook（前向钩子）**：给模型每一层挂一个回调，前向传播时自动打印「这层输入什么形状 → 输出什么形状」。跑一次样例 batch，就能得到**整个模型的形状变换地图**。

类比：像给每层加了个「探针 tap」，信号流过时自动记录。`register_forward_hook` 返回的句柄最后要 `remove()`，否则 hook 一直挂着（类似你注册中断/回调后要记得注销，避免泄漏）。

典型错误：期望 `[batch, channels, H, W]`，实际传了 `[batch, features]`（比如忘了 reshape，或 CNN 前少了通道维）。

#### ② NaN Loss（损失变 NaN）

```python
def detect_nan(model, loss, step):
    if torch.isnan(loss):
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any(): print(f"NaN grad in {name}")
                if torch.isinf(param.grad).any(): print(f"Inf grad in {name}")
```

NaN = 有东西「爆了」。常见原因：学习率太高、自定义 loss 里除以 0、对 0 或负数取 log、RNN 梯度爆炸。

这个函数的巧思：loss 变 NaN 时，**逐层扫描梯度**，定位到底哪一层先出 NaN/Inf——直接告诉你爆炸源头，而不是只知道「反正爆了」。

#### ③ Data Leakage（数据泄漏）

```python
def check_data_leakage(train_set, test_set, id_column="id"):
    overlap = set(train_set[id_column]) & set(test_set[id_column])
    if overlap:
        print(f"DATA LEAKAGE: {len(overlap)} samples in both")
```

**测试集 99% 准确率？先别高兴，多半是 bug。** 因为训练集和测试集有重叠——模型「见过答案」。用集合求交集检测重叠 ID 即可。

还有更隐蔽的**时间泄漏**：用未来数据预测过去。解法：按时间戳排序后再切分。这点你做业务系统时应该有共鸣（回测里最容易踩）。

#### ④ Wrong Device（设备放错）

```python
def check_devices(model, *tensors):
    model_device = next(model.parameters()).device
    for i, t in enumerate(tensors):
        if t.device != model_device:
            print(f"WARNING: tensor {i} on {t.device}, model on {model_device}")
```

两种表现：
- 硬报错：CPU 张量和 GPU 张量做运算 → 直接 RuntimeError（这个还算「善良」，会报）
- **沉默变慢**：某个张量偷偷留在 CPU，训练能跑但巨慢

`next(model.parameters()).device` 是「取模型所在设备」的惯用法。

---

### Part 8：TensorBoard（第 3 层，训练可视化）

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/experiment_1")

for step in range(num_steps):
    loss = train_step(model, batch)
    writer.add_scalar("loss/train", loss.item(), step)          # 标量曲线
    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], step)
    if step % 100 == 0:
        for name, param in model.named_parameters():
            writer.add_histogram(f"weights/{name}", param, step)      # 权重分布
            if param.grad is not None:
                writer.add_histogram(f"grads/{name}", param.grad, step)  # 梯度分布
writer.close()
```

`SummaryWriter` 把指标写到 `runs/` 目录，`tensorboard --logdir=runs` 起个网页看曲线。

**最实用的是这张「看曲线诊断病症」对照表**（务必记住）：

| 现象 | 病因 |
|------|------|
| loss 不降 | 学习率太低，或架构有问题 |
| loss 剧烈震荡 | 学习率太高 |
| loss 变 NaN | 数值不稳定（见 NaN 一节） |
| 训练 loss 降、验证 loss 升 | **过拟合** |
| 权重直方图塌缩到 0 | 梯度消失 |
| 梯度直方图爆炸 | 需要梯度裁剪 |

这相当于给你一套「示波器读波形 → 判断电路故障」的经验表，只不过看的是训练曲线。

---

### Part 9：VS Code / Cursor 图形化调试

```json
{
  "version": "0.2.0",
  "configurations": [{
    "name": "Debug Training",
    "type": "debugpy",
    "request": "launch",
    "program": "${file}",
    "console": "integratedTerminal",
    "justMyCode": false
  }]
}
```

`launch.json` 放到项目 `.vscode/` 下，就能点行号打断点、在 Variables 面板看张量属性、在 Debug Console 里跑任意表达式。

一个**关键设置**：`"justMyCode": false`——默认只调试你自己的代码，设为 `false` 才能**步进到 PyTorch 内部**（排查框架层问题时需要）。这套你在 Cursor 里直接能用。

---

## 四、推荐的完整工作流（Use It）

文档把工具串成了一条实战流水线，照着做就行：

1. **训练前**：`check_shapes` 跑一个样例 batch，确认每层进出维度符合预期
2. **前 10 步**：`debug_print` 打 loss / 输出 / 梯度，确认没 NaN、数值范围正常
3. **训练中**：记录 loss、学习率、梯度范数，TensorBoard 可视化
4. **出问题时**：在故障点放条件 `breakpoint()`，交互式查张量
5. **调性能时**：分别给数据加载 / forward / backward 计时；接近 OOM 就做内存分析

即：**事前查形状 → 早期查数值 → 全程记日志 → 出事下断点 → 慢了做 profile。**

---

## 五、配套脚本 `code/debug_tools.py`

这个文件把上面所有工具做成了**可直接跑的演示**，`main()` 依次跑 10 个 demo：print 调试、计时、tracemalloc、形状检查、NaN 检测、设备检查、梯度健康、GPU 显存、日志、条件断点示范。

它还很贴心地做了容错：

```python
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
```

没装 torch 也能跑非 torch 的 demo。你已经装了 torch，直接：

```bash
python phases/00-setup-and-tooling/12-debugging-and-profiling/code/debug_tools.py
```

里面还多了个文档正文没细讲、但很有用的 `check_gradient_health()`：算**总梯度范数**，并对「过大梯度」「零梯度」告警——判断梯度爆炸/消失的量化工具。

---

## 六、练习题怎么做（给你排了优先级）

按对你的收益排序：

1. **（最推荐）改 `debug_tools.py` 造一个 NaN**：在 forward 里除以 0，看 NaN 检测器抓不抓得到——最快建立「沉默 bug → 主动探测」的直觉
2. **`breakpoint()` 实操**：训练循环里下条件断点，练 `p x.shape / x.device / x.grad`
3. **cProfile 找最慢函数**：拿任意训练循环跑一遍
4. **TensorBoard 判过拟合**：跑个小训练，看训练/验证 loss 是否分叉
5. **tracemalloc 找数据管道内存大户**：结合上一课的 `data_utils.py` 练更佳

---

## 七、给你的一句话总结

这课不是「教你调试」——**调试基本功你早有**。它教的是在你已有能力上**加装三件 AI 专用探针**：

- **张量探针**（shape / dtype / device / NaN）
- **训练动态探针**（loss / 梯度 / 权重的曲线与直方图）
- **AI 专属 bug 清单**（形状、NaN、数据泄漏、设备）

所以你之前的判断很准：这是 Phase 0 里**唯一和「普通程序调试」有实质差异、值得认真过一遍**的课。建议至少把 `debug_tools.py` 跑一遍 + 做练习 1，其余等真正开始训模型（Phase 4+）时回头当手册查。

要不要我帮你把 `debug_tools.py` 在你环境里实际跑一遍，看看每个 demo 在你的 4050 上的真实输出？（那需要切到 Agent 模式执行）