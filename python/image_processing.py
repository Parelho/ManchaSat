import numpy as np
import cv2

class ImageProcessing:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def preprocess_frame(frame):
        frame = cv2.resize(frame, (256, 256))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray[..., np.newaxis]
        return np.expand_dims(gray.astype("float32") / 255.0, axis=0)

    @staticmethod
    def decode_mask(pred, frame_shape):
        mask = np.argmax(pred[0], axis=-1).astype(np.uint8)
        mask = cv2.resize(mask, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask
    
    @staticmethod
    def kernel_size_for_image(shape, frac=0.005):
        h, w = shape[:2]
        k = max(3, int(round(min(h, w) * frac)))
        if k % 2 == 0:
            k += 1
        return k

    @staticmethod
    def remove_small_components(mask, min_area):
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = np.zeros_like(mask)
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                out[labels == i] = 1
        return out

    @staticmethod
    def cluster_centers(centers, thresh):
        groups = []
        for c in centers:
            placed = False
            for g in groups:
                if any(np.hypot(c[0]-p[0], c[1]-p[1]) < thresh for p in g):
                    g.append(c)
                    placed = True
                    break
            if not placed:
                groups.append([c])
        merged = [(int(round(np.mean([p[0] for p in g]))),
                int(round(np.mean([p[1] for p in g])))) for g in groups]
        return merged

    @staticmethod
    def detect_ship_centers_from_mask(pred_mask, frame_shape,
                                    close_frac=0.005,
                                    min_area_frac=0.00002,
                                    merge_frac=0.02):
        """
        pred_mask: binary mask with ship pixels == 1
        frame_shape: frame.shape (height, width, ...)
        close_frac: morphological closing kernel = frac * min_dim
        min_area_frac: min connected-component area as fraction of image area
        merge_frac: distance fraction for merging close centroids
        Returns: centers (list of (x,y)), bboxes (list of (x,y,w,h)), cleaned_mask
        """
        binary = (pred_mask == 1).astype(np.uint8)

        # 1) Morphological closing to bridge tiny gaps
        k = ImageProcessing.kernel_size_for_image(frame_shape, close_frac)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 2) Remove tiny specks by area (auto threshold based on image size)
        img_area = frame_shape[0] * frame_shape[1]
        min_area = max(8, int(img_area * min_area_frac))  # tune min_area_frac if needed
        cleaned = ImageProcessing.remove_small_components(closed, min_area)

        # 3) Extract contours and centroids
        contours, _ = cv2.findContours(cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        bboxes = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))
            x, y, w, h = cv2.boundingRect(cnt)
            bboxes.append((x, y, w, h))

        # 4) Merge very-close centers (if a ship is still split)
        merge_dist = max(5, int(round(min(frame_shape[:2]) * merge_frac)))
        centers_merged = ImageProcessing.cluster_centers(centers, merge_dist)

        return centers_merged, bboxes, cleaned