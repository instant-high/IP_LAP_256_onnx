import cv2
import onnxruntime
import onnx
import numpy as np

'''
98kp_masker.onnx

Number of input nodes: 1
Number of output nodes: 3

Input Name: kp_input, Shape: [1, 3, 256, 256]

Output Name: kp_score, Shape: [1, 98]
Output Name: facemask, Shape: [1, 1, 256, 256]
Output Name: kp_output, Shape: [1, 196]
'''


class KP_MASK:
    def __init__(self, model_path="kps_student_masker.onnx.onnx", device='cpu'):
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        if device == 'cuda':
            providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "EXHAUSTIVE"}),"CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=providers)
        model = onnx.load(model_path)

    def facemask(self, image, regions="lower"):
        
        image = cv2.resize(image, (256, 256))
        image = image.astype(np.float32)
        image = image.transpose((2, 0, 1)) / 255
        image = np.expand_dims(image, axis=0).astype(np.float32)
                    
        mask1, mask2 = self.session.run(None, {'input': image})
    
        mask1 = np.squeeze(mask1, axis=0)  # lower
        mask2 = np.squeeze(mask2, axis=0)  # upper
        
        # -------- region selection --------
        if regions == "lower":
            mask = mask1
        elif regions == "upper":
            mask = mask2
        elif regions == "both":
            mask = np.clip(mask1 + mask2, 0, 1)
        else:
            raise ValueError(f"Unknown region: {regions}")
        
        mask = np.stack([mask]*3, axis=-1)
        
        return mask

