import numpy as np
import cv2, os, argparse
import subprocess
import shutil
from tqdm import tqdm

from utils import audio
from utils.draw_landmark_cv import draw_landmarks , DrawingSpec, FACEMESH_FULL
drawing_spec = DrawingSpec(color=(224,224,224), thickness=1, circle_radius=1)

import onnxruntime as ort
ort.set_default_logger_severity(3)
available_providers = ort.get_available_providers()

if "CUDAExecutionProvider" in available_providers:
    device = "cuda"
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print("Using CUDA")
else:
    device = "cpu"
    providers = ["CPUExecutionProvider"]
    print("Using CPU")
print ("")

if device == "cuda":
    detector_provider = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
else:
    detector_provider = ["CPUExecutionProvider"]
    

parser = argparse.ArgumentParser()
parser.add_argument('--input', '--input_template_video', type=str, default='video.mp4')
parser.add_argument('--audio', type=str, default='audio.wav')
parser.add_argument('--output', type=str, default='./result.mp4') 
#parser.add_argument('--static', type=bool, help='whether only use  the first frame for inference', default=False)
parser.add_argument('--landmark_gen_checkpoint_path', type=str, default='onnx_models/ip_lap_landmark_generator.onnx')
parser.add_argument('--renderer_checkpoint_path', type=str, default='onnx_models/ip_lap_renderer.onnx')
args = parser.parse_args()

landmark_gen_checkpoint_path = args.landmark_gen_checkpoint_path
renderer_checkpoint_path =args.renderer_checkpoint_path
input_video_path = args.input
input_audio_path = args.audio
outfile_path = args.output
    
ref_img_N = 25
Nl = 15
T = 5
mel_step_size = 16
img_size = 256
lip_index = [0, 17]

temp_dir = os.path.join(os.path.dirname(outfile_path), "temp")
os.makedirs(temp_dir, exist_ok=True)

if os.path.isfile(input_video_path) and input_video_path.split('.')[1] in ['jpg', 'png', 'jpeg']:
    args.static = True
    fps = 25
    

print("Loading ONNX models...")

# face alignment for facemesh
from alignment.retinaface import RetinaFace
from alignment.alignment import align_face 
detector = RetinaFace("onnx_models/scrfd_2.5g_bnkps.onnx", provider=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"], session_options=None)
print("Face alignment loaded")

# face masker
from kp_masker.kp_masker import KP_MASK
kp_mask = KP_MASK(model_path="onnx_models\kps_student_masker.onnx", device="cuda")
print("Facemasker loaded")

# facemesh replacement for mediapipe
facemesh_session = ort.InferenceSession("onnx_models/facemesh.onnx",providers=providers)
fm_input_name = facemesh_session.get_inputs()[0].name
print("Facemesh generator loaded")

landmark_session = ort.InferenceSession(landmark_gen_checkpoint_path,providers=providers)
print("Landmark generator loaded")

renderer_session = ort.InferenceSession(renderer_checkpoint_path,providers=providers)
print("Renderer loaded")

print("")
    
# ------------------------------------------------------------------------------------------------------------------- #


# the following is the index sequence for fical landmarks detected by mediapipe
ori_sequence_idx = [162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288,
                    361, 323, 454, 356, 389,  #
                    70, 63, 105, 66, 107, 55, 65, 52, 53, 46,  #
                    336, 296, 334, 293, 300, 276, 283, 282, 295, 285,  #
                    168, 6, 197, 195, 5,  #
                    48, 115, 220, 45, 4, 275, 440, 344, 278,  #
                    33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7,  #
                    362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382,  #
                    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,  #
                    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# the following is the connections of landmarks for drawing sketch image
FACEMESH_LIPS = frozenset([(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
                           (17, 314), (314, 405), (405, 321), (321, 375),
                           (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
                           (37, 0), (0, 267),
                           (267, 269), (269, 270), (270, 409), (409, 291),
                           (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
                           (14, 317), (317, 402), (402, 318), (318, 324),
                           (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
                           (82, 13), (13, 312), (312, 311), (311, 310),
                           (310, 415), (415, 308)])
FACEMESH_LEFT_EYE = frozenset([(263, 249), (249, 390), (390, 373), (373, 374),
                               (374, 380), (380, 381), (381, 382), (382, 362),
                               (263, 466), (466, 388), (388, 387), (387, 386),
                               (386, 385), (385, 384), (384, 398), (398, 362)])
FACEMESH_LEFT_EYEBROW = frozenset([(276, 283), (283, 282), (282, 295),
                                   (295, 285), (300, 293), (293, 334),
                                   (334, 296), (296, 336)])
FACEMESH_RIGHT_EYE = frozenset([(33, 7), (7, 163), (163, 144), (144, 145),
                                (145, 153), (153, 154), (154, 155), (155, 133),
                                (33, 246), (246, 161), (161, 160), (160, 159),
                                (159, 158), (158, 157), (157, 173), (173, 133)])
FACEMESH_RIGHT_EYEBROW = frozenset([(46, 53), (53, 52), (52, 65), (65, 55),
                                    (70, 63), (63, 105), (105, 66), (66, 107)])
FACEMESH_FACE_OVAL = frozenset([(389, 356), (356, 454),
                                (454, 323), (323, 361), (361, 288), (288, 397),
                                (397, 365), (365, 379), (379, 378), (378, 400),
                                (400, 377), (377, 152), (152, 148), (148, 176),
                                (176, 149), (149, 150), (150, 136), (136, 172),
                                (172, 58), (58, 132), (132, 93), (93, 234),
                                (234, 127), (127, 162)])
FACEMESH_NOSE = frozenset([(168, 6), (6, 197), (197, 195), (195, 5), (5, 4),
                           (4, 45), (45, 220), (220, 115), (115, 48),
                           (4, 275), (275, 440), (440, 344), (344, 278), ])
FACEMESH_CONNECTION = frozenset().union(*[
    FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYE,
    FACEMESH_RIGHT_EYEBROW, FACEMESH_FACE_OVAL, FACEMESH_NOSE
])

full_face_landmark_sequence = [*list(range(0, 4)), *list(range(21, 25)), *list(range(25, 91)),  #upper-half face
                               *list(range(4, 21)),  # jaw
                               *list(range(91, 131))]  # mouth


# ------------------------------------------------------------------------------------------------------------------- #


def get_landmarks_onnx(frame):

    h, w = frame.shape[:2]
    img_resized = cv2.resize(frame, (256, 256))
    img_rgb = img_resized[:, :, ::-1]
    img_norm = img_rgb.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(img_norm, axis=0)

    outputs = facemesh_session.run(None, {fm_input_name: input_tensor})
    
    landmarks = outputs[0].reshape(-1, 3)  # (478,3)
    # convert to normalized (MediaPipe style)
    landmarks[:, 0] /= 256.0
    landmarks[:, 1] /= 256.0
    landmarks[:, 2] /= 256.0

    return landmarks

def get_smoothened_landmarks(all_landmarks, windows_T=1):

    for i in range(len(all_landmarks)):
        if i + windows_T > len(all_landmarks):
            window = all_landmarks[len(all_landmarks) - windows_T:]
        else:
            window = all_landmarks[i: i + windows_T]
        for j in range(len(all_landmarks[i])):
            all_landmarks[i][j][1] = np.mean([frame_landmarks[j][1] for frame_landmarks in window])  # x
            all_landmarks[i][j][2] = np.mean([frame_landmarks[j][2] for frame_landmarks in window])  # y
            
    return all_landmarks
    
def process_video(model, img, size):

    bboxes, kpss = model.detect(img, (256,256), det_thresh=0.3)
    assert len(kpss) != 0, "No face detected"
    aimg, mat = align_face(img, kpss[0], crop_size=(256,256), template_key='ffhq_512') #ffhq_512 , arcface_128_v2
    
    return aimg, mat 

def summarize_landmark(edge_set):

    landmarks = set()
    for a, b in edge_set:
        landmarks.add(a)
        landmarks.add(b)

    return landmarks
    
    
# ------------------------------------------------------------------------------------------------------------------- # 
   

# pose landmarks are landmarks of the upper-half face(eyes,nose,cheek) that represents the pose information
all_landmarks_idx = summarize_landmark(FACEMESH_CONNECTION)
pose_landmark_idx = \
    summarize_landmark(FACEMESH_NOSE.union(*[FACEMESH_RIGHT_EYEBROW, FACEMESH_RIGHT_EYE,
                                             FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, ])).union(
        [162, 127, 234, 93, 389, 356, 454, 323])

# content_landmark include landmarks of lip and jaw which are inferred from audio
content_landmark_idx = all_landmarks_idx - pose_landmark_idx

# Makes a dictionary that behave like an object to represent each landmark        
class LandmarkDict(dict):
    def __init__(self, idx, x, y):
        self['idx'] = idx
        self['x'] = x
        self['y'] = y
    def __getattr__(self, name):
        try:
            return self[name]
        except:
            raise AttributeError(name)
    def __setattr__(self, name, value):
        self[name] = value


# ------------------------------------------------------------------------------------------------------------------- # 

print('Reading video frames ... from', input_video_path)
if not os.path.isfile(input_video_path):
    raise ValueError('the input video file does not exist')
elif input_video_path.split('.')[1] in ['jpg', 'png', 'jpeg']: #if input a single image for testing
    ori_background_frames = [cv2.imread(input_video_path)]
else:
    video_stream = cv2.VideoCapture(input_video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    if fps != 25:
        print(" input video fps:", fps,',converting to 25fps...')
        command = 'ffmpeg -y -i ' + input_video_path + ' -r 25 ' + '{}/temp_25fps.mp4'.format(temp_dir)
        subprocess.call(command, shell=True)
        input_video_path = '{}/temp_25fps.mp4'.format(temp_dir)
        video_stream.release()
        video_stream = cv2.VideoCapture(input_video_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
    assert fps == 25
    
    #input videos frames (includes background as well as face)
    ori_background_frames = []
    frame_idx = 0
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        ori_background_frames.append(frame)
        frame_idx = frame_idx + 1
input_vid_len = len(ori_background_frames)

# ------------------------------------------------------------------------------------------------------------------- # 

print('Extracting audio ... from', input_audio_path)
print ("")

if not input_audio_path.endswith('.wav'):
    command = 'ffmpeg -y -i {} -strict -2 {}'.format(input_audio_path, '{}/temp.wav'.format(temp_dir))
    subprocess.call(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    input_audio_path = '{}/temp.wav'.format(temp_dir)
wav = audio.load_wav(input_audio_path, 16000)
mel = audio.melspectrogram(wav)  # (H,W)   extract mel-spectrum

# read audio mel into list
# each mel chunk correspond to 5 video frames, used to generate one video frame
mel_chunks = []

# mel_idx_multiplier = 80. / fps
mel_idx_multiplier = 80. / fps
mel_chunk_idx = 0
while 1:
    start_idx = int(mel_chunk_idx * mel_idx_multiplier)
    if start_idx + mel_step_size > len(mel[0]):
        break
    mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])  # mel for generate one video frame
    mel_chunk_idx += 1
# mel_chunks = mel_chunks[:(len(mel_chunks) // T) * T]

# ------------------------------------------------------------------------------------------------------------------- # 

# detect facial landmarks using mediapipe tool
boxes = []  #bounding boxes of human face
lip_dists = [] #lip dists

# we define the lip dist(openness): distance between the  midpoints of the upper lip and lower lip
face_crop_results = []
# content landmarks include lip and jaw landmarks
all_pose_landmarks, all_content_landmarks = [], []
# store normalized (0..1) landmarks
all_landmarks_full = []  

plus_pixel = 0 # 10

for frame_idx, full_frame in enumerate(ori_background_frames):

    h, w = full_frame.shape[:2]
    aligned, M = process_video(detector, full_frame, 256)
    #cv2.imshow("aligned", aligned)
    #cv2.waitKey(1)
    
    landmarks_aligned = get_landmarks_onnx(aligned)  # normalized 0..1
    
    # convert to pixel in aligned space
    landmarks_px = landmarks_aligned.copy()
    landmarks_px[:, 0] *= 256
    landmarks_px[:, 1] *= 256
    
    # compute bbox directly in aligned space
    x_min, x_max, y_min, y_max = 999, -999, 999, -999
    
    for idx in all_landmarks_idx:
        x, y = landmarks_px[idx][:2]
    
        x_min = min(x_min, x)
        x_max = max(x_max, x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)
        
    pose_landmarks, content_landmarks = [], []
    
    for idx in range(len(landmarks_px)):
        x, y = landmarks_px[idx][:2]
    
        x_norm = (x - x_min) / (x_max - x_min)
        y_norm = (y - y_min) / (y_max - y_min)
    
        if idx in pose_landmark_idx:
            pose_landmarks.append([idx, x_norm, y_norm])
    
        if idx in content_landmark_idx:
            content_landmarks.append([idx, x_norm, y_norm])  


    M_inv = cv2.invertAffineTransform(M)
    
    landmarks_full = []
    for lm in landmarks_aligned:
        x = lm[0] * 256
        y = lm[1] * 256

        x_orig = M_inv[0,0]*x + M_inv[0,1]*y + M_inv[0,2]
        y_orig = M_inv[1,0]*x + M_inv[1,1]*y + M_inv[1,2]

        # normalize back to 0..1
        landmarks_full.append([x_orig / w, y_orig / h])

    landmarks_full = np.array(landmarks_full)
    all_landmarks_full.append(landmarks_full)
    
    # Lip dist
    dx = landmarks_full[lip_index[0], 0] - landmarks_full[lip_index[1], 0]
    dy = landmarks_full[lip_index[0], 1] - landmarks_full[lip_index[1], 1]
    dist = np.linalg.norm((dx, dy))
    lip_dists.append((frame_idx, dist))

    # Bounding box (full frame normalized)
    x_min, x_max, y_min, y_max = 1, 0, 1, 0

    for idx in all_landmarks_idx:
        x, y = landmarks_full[idx]

        x_min = min(x_min, x)
        x_max = max(x_max, x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)

    x_min = max(x_min - plus_pixel + 1/ 256, 0)
    x_max = min(x_max + plus_pixel + 2 / 256, 1)  # + 2
    y_min = max(y_min - plus_pixel + 1 / 256, 0)
    y_max = min(y_max + plus_pixel + 3 / 256, 1)  # + 1 .... add chin
    
    y1, y2 = int(y_min * h), int(y_max * h)
    x1, x2 = int(x_min * w), int(x_max * w)
    
    # (full frame facemesh and bbox)
    '''
    debug = full_frame.copy()
    for x_norm, y_norm in landmarks_full:
        x = int(x_norm * w)
        y = int(y_norm * h)
        cv2.circle(debug, (x, y), 1, (0, 0, 255), -1)
      
    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Facemesh", debug)
    cv2.waitKey(1)
    '''
           
    boxes.append([y1, y2, x1, x2])

boxes = np.array(boxes)

face_crop_results = [[image[y1:y2, x1:x2], (y1, y2, x1, x2)] for image, (y1, y2, x1, x2) in zip(ori_background_frames, boxes)]

# ------------------------------------------------------------------------------------------------------------------- # 

# detect facial landmarks
for frame_idx, full_frame in enumerate(ori_background_frames):
    h, w = full_frame.shape[:2]

    landmarks = all_landmarks_full[frame_idx]

    pose_landmarks, content_landmarks = [], []

    for idx in range(len(landmarks)):
        x = landmarks[idx, 0] * w
        y = landmarks[idx, 1] * h

        if idx in pose_landmark_idx:
            pose_landmarks.append((idx, x, y))

        if idx in content_landmark_idx:
            content_landmarks.append((idx, x, y))

    # normalize to crop
    y1, y2, x1, x2 = face_crop_results[frame_idx][1]

    pose_landmarks = [
        [idx, (x - x1) / (x2 - x1), (y - y1) / (y2 - y1)]
        for idx, x, y in pose_landmarks
    ]

    content_landmarks = [
        [idx, (x - x1) / (x2 - x1), (y - y1) / (y2 - y1)]
        for idx, x, y in content_landmarks
    ]

    all_pose_landmarks.append(pose_landmarks)
    all_content_landmarks.append(content_landmarks)
    

all_pose_landmarks = get_smoothened_landmarks(all_pose_landmarks, windows_T=1)
all_content_landmarks=get_smoothened_landmarks(all_content_landmarks,windows_T=1)

# ------------------------------------------------------------------------------------------------------------------- # 

# randomly select N_l reference landmarks for landmark transformer
dists_sorted = sorted(lip_dists, key=lambda x: x[1])
# the frame idxs sorted by lip openness
lip_dist_idx = np.asarray([idx for idx, dist in dists_sorted])

# linspace replacement
Nl_idxs = [lip_dist_idx[int(i)] for i in np.linspace(0, input_vid_len - 1, num=Nl)]

Nl_pose_landmarks, Nl_content_landmarks = [], []
for reference_idx in Nl_idxs:
    Nl_pose_landmarks.append(all_pose_landmarks[reference_idx])
    Nl_content_landmarks.append(all_content_landmarks[reference_idx])

# allocate arrays
Nl_pose = np.zeros((Nl, 2, 74), dtype=np.float32)
Nl_content = np.zeros((Nl, 2, 57), dtype=np.float32)

# fill
for idx in range(Nl):
    Nl_pose_landmarks[idx] = sorted(
        Nl_pose_landmarks[idx],
        key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0])
    )
    Nl_content_landmarks[idx] = sorted(
        Nl_content_landmarks[idx],
        key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0])
    )

    Nl_pose[idx, 0, :] = [l[1] for l in Nl_pose_landmarks[idx]]  # x
    Nl_pose[idx, 1, :] = [l[2] for l in Nl_pose_landmarks[idx]]  # y

    Nl_content[idx, 0, :] = [l[1] for l in Nl_content_landmarks[idx]]
    Nl_content[idx, 1, :] = [l[2] for l in Nl_content_landmarks[idx]]

# add batch dim
Nl_pose = np.expand_dims(Nl_pose, 0).astype(np.float32)
Nl_content = np.expand_dims(Nl_content, 0).astype(np.float32)

# ------------------------------------------------------------------------------------------------------------------- # 

# reference images

ref_img_idx = [int(lip_dist_idx[int(i)]) for i in np.linspace(0, input_vid_len - 1, num=ref_img_N)]

ref_imgs = [face_crop_results[idx][0] for idx in ref_img_idx]

ref_img_pose_landmarks, ref_img_content_landmarks = [], []
for idx in ref_img_idx:
    ref_img_pose_landmarks.append(all_pose_landmarks[idx])
    ref_img_content_landmarks.append(all_content_landmarks[idx])

ref_img_pose = np.zeros((ref_img_N, 2, 74), dtype=np.float32)
ref_img_content = np.zeros((ref_img_N, 2, 57), dtype=np.float32)

for idx in range(ref_img_N):
    ref_img_pose_landmarks[idx] = sorted(
        ref_img_pose_landmarks[idx],
        key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0])
    )
    ref_img_content_landmarks[idx] = sorted(
        ref_img_content_landmarks[idx],
        key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0])
    )

    ref_img_pose[idx, 0, :] = [l[1] for l in ref_img_pose_landmarks[idx]]
    ref_img_pose[idx, 1, :] = [l[2] for l in ref_img_pose_landmarks[idx]]

    ref_img_content[idx, 0, :] = [l[1] for l in ref_img_content_landmarks[idx]]
    ref_img_content[idx, 1, :] = [l[2] for l in ref_img_content_landmarks[idx]]

ref_img_full_face_landmarks = np.concatenate([ref_img_pose, ref_img_content], axis=2)  # (N,2,131)

# ------------------------------------------------------------------------------------------------------------------- # 

# draw sketches

ref_img_sketches = []
for frame_idx in range(ref_img_full_face_landmarks.shape[0]):
    full_landmarks = ref_img_full_face_landmarks[frame_idx]
    h, w = ref_imgs[frame_idx].shape[:2]

    drawn_sketch = np.zeros(
        (int(h * img_size / min(h, w)), int(w * img_size / min(h, w)), 3),
        dtype=np.float32
    )

    mediapipe_format_landmarks = [
        LandmarkDict(
            ori_sequence_idx[full_face_landmark_sequence[i]],
            full_landmarks[0, i],
            full_landmarks[1, i]
        )
        for i in range(full_landmarks.shape[1])
    ]

    drawn_sketch = draw_landmarks(
        drawn_sketch,
        mediapipe_format_landmarks,
        connections=FACEMESH_CONNECTION,
        connection_drawing_spec=drawing_spec
    )
    
    drawn_sketch = drawn_sketch.astype(np.uint8) ####
    drawn_sketch = cv2.resize(drawn_sketch, (img_size, img_size))
    ref_img_sketches.append(drawn_sketch)

# ------------------------------------------------------------------------------------------------------------------- # 

# convert to model format
ref_img_sketches = np.asarray(ref_img_sketches, dtype=np.float32) / 255.0
ref_img_sketches = np.expand_dims(ref_img_sketches, 0)  # (1,N,H,W,3)
ref_img_sketches = np.transpose(ref_img_sketches, (0, 1, 4, 2, 3))  # (1,N,3,H,W)

# images
ref_imgs = [cv2.resize(face.copy(), (img_size, img_size)) for face in ref_imgs]
ref_imgs = np.asarray(ref_imgs, dtype=np.float32) / 255.0
ref_imgs = np.expand_dims(ref_imgs, 0)  # (1,N,H,W,3)
ref_imgs = np.transpose(ref_imgs, (0, 1, 4, 2, 3))  # (1,N,3,H,W)

# prepare output video stream
frame_h, frame_w = ori_background_frames[0].shape[:-1]
temp_video_path = os.path.join(temp_dir, "result.mp4")
out_stream = cv2.VideoWriter(temp_video_path,cv2.VideoWriter_fourcc('m','p','4','v'),fps,(frame_w * 2, frame_h))


# -------------- generate final face image and output video ---------------------------------------------------------- # 

input_mel_chunks_len = len(mel_chunks)
input_frame_sequence = np.arange(input_vid_len).tolist()

num_of_repeat = input_mel_chunks_len // input_vid_len + 1
input_frame_sequence = input_frame_sequence + list(reversed(input_frame_sequence))
input_frame_sequence = input_frame_sequence * ((num_of_repeat + 1) // 2)

for batch_idx, batch_start_idx in tqdm(
    enumerate(range(0, input_mel_chunks_len - 2, 1)),
    total=len(range(0, input_mel_chunks_len - 2, 1))
):
    T_input_frame, T_ori_face_coordinates = [], []
    T_mel_batch, T_crop_face, T_pose_landmarks = [], [], []

    # collect batch
    for mel_chunk_idx in range(batch_start_idx, batch_start_idx + T):
        T_mel_batch.append(mel_chunks[max(0, mel_chunk_idx - 2)])

        input_frame_idx = int(input_frame_sequence[mel_chunk_idx])
        face, coords = face_crop_results[input_frame_idx]

        T_crop_face.append(face)
        T_ori_face_coordinates.append((face, coords))
        T_pose_landmarks.append(all_pose_landmarks[input_frame_idx])
        T_input_frame.append(ori_background_frames[input_frame_idx].copy())

    T_mels = np.asarray(T_mel_batch, dtype=np.float32)
    T_mels = np.expand_dims(T_mels, axis=(0, 2))  # (1,T,1,h,w)


    # pose landmarks
    T_pose = np.zeros((T, 2, 74), dtype=np.float32)

    for idx in range(T):
        T_pose_landmarks[idx] = sorted(
            T_pose_landmarks[idx],
            key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0])
        )

        T_pose[idx, 0, :] = [l[1] for l in T_pose_landmarks[idx]]
        T_pose[idx, 1, :] = [l[2] for l in T_pose_landmarks[idx]]

    T_pose = np.expand_dims(T_pose, 0)  # (1,T,2,74)


    # ONNX inference
    landmark_inputs = {
        "T_mels": T_mels.astype(np.float32),
        "T_pose": T_pose.astype(np.float32),
        "Nl_pose": Nl_pose,
        "Nl_content": Nl_content
    }

    predict_content = landmark_session.run(
        ["predict_content"],
        landmark_inputs
    )[0]  # (T,2,57)

    T_pose_np = T_pose.reshape(-1, 2, 74)  # (T,2,74)

    T_predict_full_landmarks = np.concatenate(
        [T_pose_np, predict_content],
        axis=2
    )  # (T,2,131)

    # draw sketches
    T_target_sketches = []

    for frame_idx in range(T):
        full_landmarks = T_predict_full_landmarks[frame_idx]
        h, w = T_crop_face[frame_idx].shape[:2]

        drawn_sketch = np.zeros(
            (int(h * img_size / min(h, w)), int(w * img_size / min(h, w)), 3),
            dtype=np.float32
        )

        mediapipe_format_landmarks = [
            LandmarkDict(
                ori_sequence_idx[full_face_landmark_sequence[i]],
                full_landmarks[0, i],
                full_landmarks[1, i]
            )
            for i in range(full_landmarks.shape[1])
        ]

        drawn_sketch = draw_landmarks(
            drawn_sketch,
            mediapipe_format_landmarks,
            connections=FACEMESH_CONNECTION,
            connection_drawing_spec=drawing_spec
        )
        
        drawn_sketch = drawn_sketch.astype(np.uint8) ###
        drawn_sketch = cv2.resize(drawn_sketch, (img_size, img_size))

        if frame_idx == 2:
            show_sketch = cv2.resize(drawn_sketch, (frame_w, frame_h)).astype(np.uint8)
            
        #cv2.imshow("Sketch",drawn_sketch)
        
        T_target_sketches.append(drawn_sketch / 255.0)

    T_target_sketches = np.asarray(T_target_sketches, dtype=np.float32)  # (T,H,W,3)
    T_target_sketches = np.transpose(T_target_sketches, (0, 3, 1, 2))   # (T,3,H,W)
    target_sketches = np.expand_dims(T_target_sketches, 0)               # (1,T,3,H,W)

    ori_face_img = cv2.resize(T_crop_face[2], (img_size, img_size)).astype(np.float32) / 255.0
    ori_face_copy = ori_face_img.copy()
    #cv2.imshow("Orig", ori_face_img)
    
    ori_face_img = np.transpose(ori_face_img, (2, 0, 1))  # (3,H,W)
    ori_face_img = np.expand_dims(ori_face_img, axis=(0, 1))  # (1,1,3,H,W)
            
    # render the full face
    renderer_inputs = {
        "face_frame_img": ori_face_img.astype(np.float32),
        "target_sketches": target_sketches.astype(np.float32),
        "ref_N_frame_img": ref_imgs.astype(np.float32),
        "ref_N_frame_sketch": ref_img_sketches.astype(np.float32),
        "audio_mels": T_mels[:, 2:3].astype(np.float32)
    }
    
    outputs = renderer_session.run(None, renderer_inputs)
    generated_face = outputs[0]
    
    gen_face = (generated_face[0].transpose(1,2,0) * 255).astype(np.uint8)

    y1, y2, x1, x2 = T_ori_face_coordinates[2][1]  # coordinates of face bounding box
    original_background = T_input_frame[2].copy()

    # lower-half masked face
    mask = kp_mask.facemask(gen_face, "lower")
    mask = cv2.rectangle(mask, (3, 3), (253, 253), (0, 0, 0), 6)
    mask = cv2.GaussianBlur(mask, (7, 7),0)
    blended = (ori_face_copy.astype(np.float32) * (1 - mask) +(gen_face.astype(np.float32) / 255.0) * mask)
    blended_uint8 = (blended * 255).clip(0,255).astype(np.uint8)
    blended_resized = cv2.resize(blended_uint8, (x2 - x1, y2 - y1))
    
    T_input_frame[2][y1:y2, x1:x2] = blended_resized
    full = T_input_frame[2].astype(np.uint8)
    
    # final output
    full = np.concatenate([show_sketch, full], axis=1)
   
    out_stream.write(full)
    cv2.imshow("Press 'Esc' to stop",full)
    
    k = cv2.waitKey(1)
    if k == 27:    
        cv2.destroyAllWindows()
        out_stream.release()
        break
        
    if batch_idx == 0:
        out_stream.write(full)

out_stream.release()

print("Temporary video saved to:", os.path.join(temp_dir, "result.mp4"))


# ------------------------------------------------------------------------------------------------------------------- #


# FFmpeg command for audio muxing
video_path = temp_video_path
video_abs = os.path.abspath(video_path)
audio_abs = os.path.abspath(input_audio_path)
outfile_abs = os.path.abspath(outfile_path)

command = f'ffmpeg -y -i "{video_abs}" -i "{audio_abs}" -c:v copy -c:a aac -shortest "{outfile_abs}"'
print("Running FFmpeg command:")

result = subprocess.run(command, shell=True, capture_output=True, text=True)
print(result.stdout)
print(result.stderr)

if result.returncode != 0:
    print("Audio muxing failed!")
else:
    print("Audio muxing succeeded:", outfile_abs)
    print("done")
    
shutil.rmtree(temp_dir, ignore_errors=True)

