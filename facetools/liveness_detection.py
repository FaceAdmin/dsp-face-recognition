import urllib
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
import progressbar
from PIL import Image
from torchvision import transforms as T

pbar = None

def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None

class LivenessDetector:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            print("Downloading the DeepPixBiS onnx checkpoint...")
            url = "https://github.com/ffletcherr/face-recognition-liveness/releases/download/v0.1/OULU_Protocol_2_model_0_0.onnx"
            urllib.request.urlretrieve(
                url,
                self.model_path.absolute().as_posix(),
                show_progress
            )

        self.session = onnxruntime.InferenceSession(
            self.model_path.as_posix(), providers=["CPUExecutionProvider"]
        )

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_liveness_score(self, face_bgr: np.ndarray) -> float:
        """
        Принимает фрагмент лица (BGR) и возвращает float (0..1).
        Чем выше значение, тем "живее" лицо.
        """
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)
        tensor_img = self.transform(pil_img).unsqueeze(0).numpy()

        output_pixel, output_binary = self.session.run(
            ["output_pixel", "output_binary"],
            {"input": tensor_img.astype(np.float32)}
        )

        liveness_score = (np.mean(output_pixel) + np.mean(output_binary)) / 2.0
        return float(liveness_score)
