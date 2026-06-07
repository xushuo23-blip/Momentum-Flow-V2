from ml_collections import ConfigDict


def get_config():
    cfg = ConfigDict()

    # ---- prompt 来源 ----
    cfg.prompt_file = "./Data/sample_dataset/example_prompts/prompts.json"
    cfg.prompts = ()                          # 命令行传入 prompt 时留空

    # ---- checkpoint ----
    cfg.checkpoint = "./Model/checkpoints/base_train/checkpoint.200.pt"
    cfg.train_config = "./configs/base_training.py"

    # ---- 采样控制 ----
    cfg.batch_size = 16                       # 生成图像数量
    cfg.steps = 100                           # 反向积分步数
    cfg.tau = 0.1                             # None: 使用训练 kinetic 中的 tau_max
    cfg.eta = 1                               # 0: 确定性 ODE; 1: 全随机 SDE

    # ---- 文本编码器 (编码 prompt, 与训练一致) ----
    cfg.text_encoder = ConfigDict()
    cfg.text_encoder.enabled = True
    cfg.text_encoder.model_name = "openai/clip-vit-large-patch14"
    cfg.text_encoder.max_length = 77
    cfg.text_encoder.return_tokens = True

    # ---- VAE decoder (latent → RGB, 与训练一致) ----
    # SD3 medium VAE: vae_scale_factor=8, latent_channels=16
    cfg.latent = ConfigDict()
    cfg.latent.enabled = True
    cfg.latent.vae_model = "stabilityai/stable-diffusion-3-medium-diffusers"
    cfg.latent.subfolder = "vae"
    cfg.latent.dtype = "float32"

    # ---- 输出 ----
    cfg.out = "./Sampling/outputs/samples.png"

    return cfg
