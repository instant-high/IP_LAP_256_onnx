# IP_LAP_256_onnx
IP_LAP: Identity-Preserving Talking Face Generation with Landmark and Appearance Priors - ONNX implementation  
Code mainly based on https://github.com/langzizhixin/IP_LAP_256  

  Update April 02. 2026:  
    Changed onnx conversion back to fixed input shapes (dynamic not working without modifying original model)  
    New onnx models available for testing:  
    Nref_25.onnx .... Nref_3.onnx (4 x faster inference, lower quality)  
      Changed inference code, ...Nref_3 onnx as default.  
      You have to set ref_img_N = 25 manually in the script to try the ...Nref_25 model
    

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
  


