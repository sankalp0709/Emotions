import json
import numpy as np
import cv2

class EmotionModelAdapter:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        self.model_type = cfg.get('model_type')
        self.model_path = cfg.get('model_path')
        self.labels = cfg.get('labels', [])
        self.label_map_path = cfg.get('label_map_path')
        self.label_map = None
        if self.label_map_path:
            try:
                with open(self.label_map_path, 'r', encoding='utf-8') as lf:
                    lm = json.load(lf)
                self.label_map = {str(k): v for k, v in lm.items()}
            except Exception:
                self.label_map = None
        size = cfg.get('input_size', [64, 64])
        self.input_w, self.input_h = int(size[0]), int(size[1])
        self.color = cfg.get('color_mode', 'rgb')
        self.normalize = cfg.get('normalize', True)
        self.grayscale = cfg.get('grayscale', False)
        self._model = None
        if self.model_type == 'onnx':
            import onnxruntime as ort
            self._sess = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            self._input_name = self._sess.get_inputs()[0].name
            self._output_name = self._sess.get_outputs()[0].name
            self._input_shape = self._sess.get_inputs()[0].shape
        elif self.model_type == 'tflite':
            import tensorflow as tf
            self._tflite = tf.lite.Interpreter(model_path=self.model_path)
            self._tflite.allocate_tensors()
            self._input_details = self._tflite.get_input_details()
            self._output_details = self._tflite.get_output_details()
        elif self.model_type == 'savedmodel':
            import tensorflow as tf
            self._model = tf.saved_model.load(self.model_path)

    def _prep(self, img: np.ndarray) -> np.ndarray:
        img = cv2.resize(img, (self.input_w, self.input_h))
        img = img.astype('float32')
        if self.normalize:
            img = img / 255.0
        x = np.expand_dims(img, 0)
        if self.model_type == 'onnx' and self._input_shape is not None and len(self._input_shape) == 4:
            # Handle NCHW models: [1,3,224,224]
            if (self._input_shape[1] == 3 and self._input_shape[2] == self.input_w):
                x = np.transpose(x, (0, 3, 1, 2))
        return x

    def predict(self, img: np.ndarray):
        x = self._prep(img)
        if self.model_type == 'onnx':
            y = self._sess.run([self._output_name], {self._input_name: x})[0]
        elif self.model_type == 'tflite':
            self._tflite.set_tensor(self._input_details[0]['index'], x)
            self._tflite.invoke()
            y = self._tflite.get_tensor(self._output_details[0]['index'])
        elif self.model_type == 'savedmodel':
            y = self._model(x).numpy()
        else:
            return None, None
        probs = y[0]
        # Apply softmax if outputs aren't normalized
        s = float(np.sum(probs))
        if not np.isfinite(s) or abs(s - 1.0) > 1e-3:
            exp = np.exp(probs - np.max(probs))
            probs = exp / np.sum(exp)
        idx = int(np.argmax(probs))
        if self.label_map is not None:
            label = self.label_map.get(str(idx), str(idx))
        else:
            label = self.labels[idx] if 0 <= idx < len(self.labels) else str(idx)
        score = float(probs[idx])
        return label, score

    def predict_with_proba(self, img: np.ndarray):
        x = self._prep(img)
        if self.model_type == 'onnx':
            y = self._sess.run([self._output_name], {self._input_name: x})[0]
        elif self.model_type == 'tflite':
            self._tflite.set_tensor(self._input_details[0]['index'], x)
            self._tflite.invoke()
            y = self._tflite.get_tensor(self._output_details[0]['index'])
        elif self.model_type == 'savedmodel':
            y = self._model(x).numpy()
        else:
            return None, None, {}
        probs = y[0]
        s = float(np.sum(probs))
        if not np.isfinite(s) or abs(s - 1.0) > 1e-3:
            exp = np.exp(probs - np.max(probs))
            probs = exp / np.sum(exp)
        idx = int(np.argmax(probs))
        names = []
        if self.label_map is not None:
            for i in range(len(probs)):
                names.append(self.label_map.get(str(i), str(i)))
        else:
            for i in range(len(probs)):
                if 0 <= i < len(self.labels):
                    names.append(self.labels[i])
                else:
                    names.append(str(i))
        label = names[idx]
        score = float(probs[idx])
        proba = {names[i]: float(probs[i]) for i in range(len(probs))}
        return label, score, proba
