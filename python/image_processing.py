import numpy as np
import cv2

class ImageProcessing:
    def __init__(self) -> None:
        pass

    def preprocess_frame(frame):
        frame = cv2.resize(frame, (256, 256))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray[..., np.newaxis]
        return np.expand_dims(gray.astype("float32") / 255.0, axis=0)

    def decode_mask(pred, frame_shape):
        mask = np.argmax(pred[0], axis=-1).astype(np.uint8)
        mask = cv2.resize(mask, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask