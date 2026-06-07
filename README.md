# OurMomentumFlow

用于研究 **text-conditioned Momentum Flow** 的代码工程。当前版本在 VAE latent 空间训练 Momentum Flow，使用 MMDiT-lite 双流 Transformer 主干。

两个可训练网络：

- `r_theta`：端点方向网络，学习 latent 空间端点方向。
- `s_theta_v`：velocity-space score 网络，修正反向 kinetic dynamics。

条件来自冻结 CLIP text encoder（CLIP-L/14），输出 pooled embedding + token sequence 进入 MMDiT 的 image/text 双流 attention。图像端使用冻结 SD3 medium VAE，训练和生成均在 latent 空间进行。

---

## Training/ 和 Sampling/ 是两个独立工程

**`Training/` 目录只负责训练**，不参与后续生成。它依赖 `Data/`、`Model/`、`configs/`。

**`Sampling/` 目录只负责从 checkpoint 生成图像**，不依赖 `Training/` 目录的任何代码。`Sampling/` 内部自带训练阶段用到的部分模块副本：

```text
Sampling/
  text_encoder.py    ← 冻结 CLIP text encoder (独立副本)
  latent_vae.py      ← 冻结 VAE decoder (独立副本)
  schedules.py       ← kinetic schedule 数学工具 (独立副本)
  reverse.py         ← reverse kinetic ODE/SDE 积分器
  generation.py      ← 编排层：加载 checkpoint → 重建网络 → 采样 → VAE decode
```

这样设计的好处是：**只需 `configs/`、`Data/sample_dataset/`、`Model/`、`Sampling/` 四个目录，就能从训练好的 checkpoint 独立生成图像**，不需要安装训练依赖或保留训练代码。

---

## 环境依赖

```bash
pip install -r requirements.txt
```

主要依赖：`torch`、`torchvision`、`transformers`、`diffusers`、`ml-collections`、`wandb`、`pillow`

---

## 目录结构

```text
OurMomentumFlow/
  Data/
    train_dataset/       训练图像和 prompt 数据
    sample_dataset/      生成/采样时使用的 prompt 文件
  Training/              训练管线（独立于 Sampling）
    train.py             训练入口
    text_encoder.py      冻结 CLIP text encoder
    latent_vae.py        冻结 VAE encoder/decoder
    schedules.py         kinetic schedule 数学工具
    state_sampling.py    训练状态采样 (z_t, v_t)
    loss.py              两个 loss 的计算
    preview_sampling.py  训练中轻量预览生成
  Sampling/              生成管线（独立于 Training）
    generation.py        编排层：加载 checkpoint → 采样 → VAE decode
    reverse.py           reverse kinetic ODE/SDE 积分器
    text_encoder.py      冻结 CLIP text encoder (独立副本)
    latent_vae.py        冻结 VAE decoder (独立副本)
    schedules.py         kinetic schedule 数学工具 (独立副本)
  Model/
    networks/
      mmdit.py           MomentumMMDiT (MMDiT-lite 双流 Transformer)
      factory.py         网络构造入口
    checkpoints/         训练得到的 checkpoint
  configs/               配置文件和加载工具
    base_training.py     基础训练配置
    base_sampling.py     基础采样配置
    exp1/                experiment 1 专用配置
  experiments/           实验入口目录
  requirements.txt       Python 依赖
```

---

## VAE latent 空间说明

当前使用 SD3 medium VAE（冻结）：

- `vae_scale_factor = 8` （每边压缩 8 倍）
- `latent_channels = 16`

`dataset.image_size=256` 时：RGB `256×256×3` → VAE encode → latent `32×32×16`。代码自动推断 `model.image_size=32`、`model.out_channels=16`、`model.in_channels=32`。

训练时 VAE encoder 把 `x0` 编码为 `z0`，在 latent 空间构造 Momentum Flow 训练状态。生成时 reverse kinetic 采样器在 latent 空间中输出 `ẑ0`，再由 VAE decoder 还原为 RGB 图像。

---

## 配置文件

所有实验参数放在 `configs/`。做新实验时复制一份 base config 修改即可，不把超参数写死在脚本里。具体实验配置放在子目录（如 `configs/exp1/`）。

---

## 训练

```bash
python Training/train.py --config configs/base_training.py
```

每个 step 流程：

1. 从 `Data/train_dataset/` 读取图像和 prompt
2. VAE encoder 将 RGB 编码为 latent `z0`
3. 冻结 CLIP text encoder 编码 prompt → `e_p` (pooled) + `C_0` (tokens)
4. 构造 Momentum Flow 相空间训练状态 `(z_t, v_t)`
5. 在同一批状态上计算两个 loss（双 optimizer 独立 backward + clip + step）：
   - `loss_r`：端点方向 MSE（只经过 `r_net`）
   - `loss_s`：NCSN 风格加权 velocity score MSE（只经过 `score_net`）

checkpoint 保存：当前 step、训练 config、`r_net` 权重、`score_net` 权重。

---

## 采样 / 生成

采样只需 checkpoint 和 prompt。核心接口在 `Sampling/generation.py`：

```python
from Sampling.generation import generate_samples_from_checkpoint

samples = generate_samples_from_checkpoint(
    "Model/checkpoints/base_train/checkpoint.1000.pt",
    prompts=["a cat sitting on a sofa"],
    batch_size=16,
    steps=100,
    tau=0.1,
    eta=1.0,
)
```

流程：加载 checkpoint → 按训练 config 重建 MMDiT 网络和 text encoder → 编码 prompt → reverse kinetic ODE/SDE 在 latent 空间迭代 → VAE decoder 还原为 RGB。

`Sampling/` 不依赖 `Training/`。只保留 `configs/`、`Data/sample_dataset/`、`Model/`、`Sampling/` 四个目录即可独立生成。

---

## wandb 使用方式

默认上传到 `entity=MAIR_HUST, project=Momentum-Flow`。API key 不写进代码。若 `wandb.mode="online"`，先执行 `wandb login` 或设置环境变量 `WANDB_API_KEY`。不想启用 wandb：

```bash
python Training/train.py --config configs/base_training.py --no-wandb
```

---

## Git 注意事项

`.gitignore` 已忽略：`.DS_Store`、Python 缓存、虚拟环境、wandb 日志、`Model/checkpoints/`、`Sampling/outputs/`、大数据目录。
