# Prompt-image dataset

训练数据使用 `prompt_image` 格式：每条样本包含一张图片和一个 prompt。每个数据集都放在 `Data/train_dataset/<dataset_name>/` 下，数据集目录中必须有一个 `samples.json`。

```text
Data/train_dataset/
  example_prompt_image/
    samples.json
    images/
      sample_00.png
      sample_01.png
  celeba_prompt_image/
    samples.json
    img_align_celeba/
      000001.jpg
      000002.jpg
```

`<dataset_name>/samples.json`:

```json
[
  {"image": "images/sample_00.png", "prompt": "your first prompt"},
  {"image": "images/sample_01.png", "prompt": "your second prompt"}
]
```

其中 `image` 是相对于该数据集目录的路径。例如当前 CelebA 数据集：

```json
[
  {"image": "img_align_celeba/000001.jpg", "prompt": "a portrait photo of a human face"}
]
```

当前可用数据集：

- `example_prompt_image/`：极小示例数据，用于检查格式。
- `celeba_prompt_image/`：CelebA aligned face 数据，包含 `202599` 张图片和同等数量的 manifest 记录。

原始图片可以有不同分辨率。训练时 `build_image_transform(image_size)` 会统一执行 resize、center crop、tensor 化和归一化。
