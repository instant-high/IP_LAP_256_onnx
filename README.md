# IP_LAP_256_onnx
IP_LAP: Identity-Preserving Talking Face Generation with Landmark and Appearance Priors - ONNX implementation  
Code mainly based on https://github.com/langzizhixin/IP_LAP_256  

  Changes:

  * Removed torch dependencies
  * Removed mediapipe dependencies
  * Converted IP-LAP torch models to onnx
  * ONNX Retinaface for face-detection and alignment
  * ONNX Facemesh model as mediapipe replacement
  * ONNX facemask model

  Tested on windows (cpu and cuda - nvidia rtx3060)  
  This is just a POC  

  Python 3.10    
  FFmpeg required  

  Download all models from releases to /onnx_models
  
  Run inference:  
  python inference_256.py --input "video.mp4" --audio "audio.wav" --output "result.mp4"  
  (CPU inference is quite slow, so nvidia GPU recommended)
  


