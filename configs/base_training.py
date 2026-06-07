from ml_collections import ConfigDict


def get_config():
    cfg = ConfigDict()

    cfg.dataset = ConfigDict()
    cfg.dataset.name = "prompt_image"
    cfg.dataset.folder = "./Data/train_dataset/example_prompt_image"
    cfg.dataset.manifest = "samples.json"
    cfg.dataset.image_size = 256
    cfg.dataset.batch_size = 64
    cfg.dataset.num_workers = 4
    cfg.dataset.drop_last = True

    cfg.model = ConfigDict()
    cfg.model.name = "momentum_mmdit"
    cfg.model.patch_size = 2
    cfg.model.dim = 512
    cfg.model.depth = 12
    cfg.model.heads = 8
    cfg.model.mlp_ratio = 4.0
    cfg.model.multiple_of = 256
    cfg.model.dropout = 0.0

    cfg.text_encoder = ConfigDict()
    cfg.text_encoder.enabled = True
    cfg.text_encoder.model_name = "openai/clip-vit-large-patch14"
    cfg.text_encoder.max_length = 77
    cfg.text_encoder.return_tokens = True

    # ---- VAE latent space 配置 ----
    # 使用 SD3 medium VAE (冻结，不参与训练):
    #   vae_scale_factor = 8   (空间每边压缩 8 倍，由 4 层下采样 block 决定)
    #   latent_channels  = 16  (latent 通道数)
    #
    # 则 dataset.image_size=256 时:
    #   RGB 256×256×3  --VAE encode-->  latent 32×32×16
    #   latent_image_size = 256 / 8 = 32
    cfg.latent = ConfigDict()
    cfg.latent.enabled = True
    cfg.latent.vae_model = "stabilityai/stable-diffusion-3-medium-diffusers"
    cfg.latent.subfolder = "vae"
    cfg.latent.sample = False         # False: VAE encoder 取 posterior mode (确定性)
    cfg.latent.dtype = "float32"      # VAE 内部计算精度

    cfg.kinetic = ConfigDict()
    cfg.kinetic.lambda_const = 2.0
    cfg.kinetic.rho = 2.0
    cfg.kinetic.tau_min = 0.05
    cfg.kinetic.tau_max = 0.20
    cfg.kinetic.num_quad = 128

    cfg.optimizer_r = ConfigDict()
    cfg.optimizer_r.name = "adamw"
    cfg.optimizer_r.lr = 1e-4
    cfg.optimizer_r.betas = (0.9, 0.95)
    cfg.optimizer_r.weight_decay = 0.03
    cfg.optimizer_r.eps = 1e-8

    cfg.optimizer_s = ConfigDict()
    cfg.optimizer_s.name = "adamw"
    cfg.optimizer_s.lr = 1e-4
    cfg.optimizer_s.betas = (0.9, 0.95)
    cfg.optimizer_s.weight_decay = 0.03
    cfg.optimizer_s.eps = 1e-8

    cfg.train = ConfigDict()
    cfg.train.steps = 100000
    cfg.train.grad_clip_norm = 1.0
    cfg.train.log_every = 10
    cfg.train.save_every = 1000
    cfg.train.keep_last_checkpoints = 5
    cfg.train.output_dir = "./Model/checkpoints/base_train"

    cfg.preview = ConfigDict()
    cfg.preview.enabled = True
    cfg.preview.every = 1000
    cfg.preview.batch_size = 4
    cfg.preview.steps = 50
    cfg.preview.tau = 0.10
    cfg.preview.eta = 1

    cfg.wandb = ConfigDict()
    cfg.wandb.enabled = True
    cfg.wandb.project = "Momentum-Flow"
    cfg.wandb.entity = "MAIR_HUST"
    cfg.wandb.run_name = "base_train"
    cfg.wandb.mode = "online"
    cfg.wandb.dir = "./wandb"
    cfg.wandb.tags = ()
    cfg.wandb.notes = ""

    return cfg
