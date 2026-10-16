from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ReIDEncoder:
    """OSNet encoder loaded from the official torchreid package and a real checkpoint."""

    def __init__(self, model_name: str, checkpoint: str | Path, device: str = "auto") -> None:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Không thấy checkpoint Person Re-ID: {checkpoint_path}. "
                "Chạy scripts/fetch_reid_model.py hoặc chỉ định checkpoint hợp lệ."
            )
        try:
            import torch
            import torchreid
        except ImportError as exc:
            raise RuntimeError("Thiếu torch/torchreid; cài extra reid theo README") from exc
        self.torch = torch
        selected = "cuda" if device == "auto" and torch.cuda.is_available() else (
            "cpu" if device == "auto" else device
        )
        self.device = torch.device(selected)
        self.model = torchreid.models.build_model(model_name, num_classes=1, pretrained=False)
        torchreid.utils.load_pretrained_weights(self.model, str(checkpoint_path))
        self.model.to(self.device).eval()

    def encode(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        if not images_bgr:
            return np.empty((0, 0), dtype=np.float32)
        tensors = []
        for image in images_bgr:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (128, 256), interpolation=cv2.INTER_LINEAR)
            tensor = self.torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
            mean = self.torch.tensor([0.485, 0.456, 0.406])[:, None, None]
            std = self.torch.tensor([0.229, 0.224, 0.225])[:, None, None]
            tensors.append((tensor - mean) / std)
        batch = self.torch.stack(tensors).to(self.device)
        with self.torch.inference_mode():
            output = self.model(batch)
            output = self.torch.nn.functional.normalize(output, p=2, dim=1)
        return output.detach().cpu().numpy().astype(np.float32)

    def aggregate(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        embeddings = self.encode(images_bgr)
        if len(embeddings) == 0:
            raise ValueError("Không có crop hợp lệ để tạo embedding")
        vector = embeddings.mean(axis=0)
        norm = np.linalg.norm(vector)
        if norm <= 1e-12:
            raise ValueError("Embedding rỗng")
        return (vector / norm).astype(np.float32)

