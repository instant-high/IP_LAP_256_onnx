import cv2
import numpy as np

warp_templates = {
    'arcface_112_v1': np.array([
        [0.35473214, 0.45658929],
        [0.64526786, 0.45658929],
        [0.50000000, 0.61154464],
        [0.37913393, 0.77687500],
        [0.62086607, 0.77687500]
    ]),
    'arcface_112_v2': np.array([
        [0.34191607, 0.46157411],
        [0.65653393, 0.45983393],
        [0.50022500, 0.64050536],
        [0.37097589, 0.82469196],
        [0.63151696, 0.82325089]
    ]),
    'arcface_128_v2': np.array([
        [0.36167656, 0.40387734],
        [0.63696719, 0.40235469],
        [0.50019687, 0.56044219],
        [0.38710391, 0.72160547],
        [0.61507734, 0.72034453]
    ]),
    'ffhq_512': np.array([
        [0.37691676, 0.46864664],
        [0.62285697, 0.46912813],
        [0.50123859, 0.61331904],
        [0.39308822, 0.72541100],
        [0.61150205, 0.72490465]
    ]),
    'mtcnn_512': np.array([
        [0.36562865, 0.46733799],
        [0.63305391, 0.46585885],
        [0.50019127, 0.61942959],
        [0.39032951, 0.77598822],
        [0.61178945, 0.77476328]
    ]),
    'styleganex_512': np.array([
        [0.43907768, 0.54098284],
        [0.56204778, 0.54122359],
        [0.50123859, 0.61331904],
        [0.44716341, 0.66936502],
        [0.55637032, 0.66911184]
    ]),
    'wav2lip': np.array([
        [0.322946, 0.326963],
        [0.675318, 0.325014],
        [0.500252, 0.617366],
        [0.355493, 0.803655],
        [0.647299, 0.802041]
    ])            
    
}
    
def estimate_matrix(face_landmarks, crop_size, template_key='arcface_128_v2'):

    if template_key not in warp_templates:
        raise ValueError(f"Template key '{template_key}' not found in warp templates.")
    
    normed_template = warp_templates[template_key] * crop_size
    affine_matrix, _ = cv2.estimateAffinePartial2D(face_landmarks, normed_template, method=cv2.RANSAC, ransacReprojThreshold=100)
    return affine_matrix

    
def align_face_wav_2_lip(image, face_landmarks, crop_size=(256, 384), template_key='wav2lip',extend=(16,16)):  # (horizontal, vertical) 12, 20
    # crop_size 3:4
    # Unpack extensions
    
    extend_x, extend_y = extend

    # Original size
    crop_w, crop_h = crop_size

    # New extended size
    new_w = crop_w + 2 * extend_x
    new_h = crop_h + 2 * extend_y

    # Get original affine matrix
    affine_matrix = estimate_matrix(face_landmarks, crop_size, template_key)

    # Modify affine to shift face into new padded canvas
    shift_matrix = np.array([[1, 0, extend_x],
                             [0, 1, extend_y]], dtype=np.float32)
    extended_affine = shift_matrix @ np.vstack([affine_matrix, [0, 0, 1]])  # shape (2,3)

    # Apply warp with extended output size
    aligned_face = cv2.warpAffine(image, extended_affine[:2], (new_w, new_h),
                                  flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)

    return aligned_face, extended_affine[:2]

    
def align_face(image, face_landmarks, crop_size=(128, 128), template_key='arcface_128_v2'):

    affine_matrix = estimate_matrix(face_landmarks, crop_size, template_key)
    aligned_face = cv2.warpAffine(image, affine_matrix, crop_size, flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)

    return aligned_face, affine_matrix

