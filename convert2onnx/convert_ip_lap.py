
# dynamic N_ref renderer input

import sys
import os
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from models.landmark_generator import Landmark_generator, Fusion_transformer_encoder
from models.video_renderer import DenseFlowNetwork, TranslationNetwork


class ManualMultiheadAttention(nn.Module):
	def __init__(self, mha):
		super().__init__()
		self.num_heads = mha.num_heads
		self.head_dim = mha.head_dim
		self.embed_dim = mha.embed_dim
		self.in_proj_weight = mha.in_proj_weight
		self.in_proj_bias = mha.in_proj_bias
		self.out_proj = mha.out_proj
		self.scale = self.head_dim ** -0.5

	def forward(self, x):
		B, S, E = x.shape
		qkv = torch.nn.functional.linear(x, self.in_proj_weight, self.in_proj_bias)
		qkv = qkv.reshape(B, S, 3, self.num_heads, self.head_dim)
		qkv = qkv.permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]
		attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
		attn = torch.softmax(attn, dim=-1)
		out = torch.matmul(attn, v)
		out = out.transpose(1, 2).reshape(B, S, E)
		out = self.out_proj(out)
		return out


class UnfusedTransformerEncoderLayer(nn.Module):
	def __init__(self, layer):
		super().__init__()
		self.self_attn = ManualMultiheadAttention(layer.self_attn)
		self.linear1 = layer.linear1
		self.linear2 = layer.linear2
		self.norm1 = layer.norm1
		self.norm2 = layer.norm2
		self.activation = layer.activation

	def forward(self, src):
		src2 = self.self_attn(src)
		src = src + src2
		src = self.norm1(src)
		src2 = self.linear2(self.activation(self.linear1(src)))
		src = src + src2
		src = self.norm2(src)
		return src


class UnfusedTransformerEncoder(nn.Module):
	def __init__(self, encoder):
		super().__init__()
		self.layers = nn.ModuleList([
			UnfusedTransformerEncoderLayer(layer) for layer in encoder.layers
		])

	def forward(self, src):
		output = src
		for layer in self.layers:
			output = layer(output)
		return output


def load_checkpoint(model, path):
	checkpoint = torch.load(path, map_location='cpu')
	s = checkpoint['state_dict']
	new_s = {}
	for k, v in s.items():
		new_k = k.replace('module.', '', 1) if k.startswith('module') else k
		new_s[new_k] = v
	model.load_state_dict(new_s, strict=False)
	return model.eval()


class LandmarkGeneratorInference(nn.Module):
	def __init__(self, model):
		super().__init__()
		self.mel_encoder = model.mel_encoder
		self.ref_encoder = model.ref_encoder
		self.pose_encoder = model.pose_encoder
		self.Norm = model.Norm
		self.mouse_keypoint_map = model.mouse_keypoint_map
		self.jaw_keypoint_map = model.jaw_keypoint_map

		self.position_v = model.fusion_transformer.position_v
		self.position_a = model.fusion_transformer.position_a
		self.modality = model.fusion_transformer.modality
		self.transformer_encoder = UnfusedTransformerEncoder(model.fusion_transformer.transformer_encoder)

	def forward(self, T_mels, T_pose, Nl_pose, Nl_content):
		B = 1
		T = 5
		N_l = 15

		Nl_ref = torch.cat([Nl_pose, Nl_content], dim=3)
		Nl_ref = Nl_ref.reshape(B * N_l, 2, 131)

		T_mels_flat = T_mels.reshape(B * T, 1, 80, 16)
		T_pose_flat = T_pose.reshape(B * T, 2, 74)

		mel_embedding = self.mel_encoder(T_mels_flat).squeeze(-1).squeeze(-1)
		pose_embedding = self.pose_encoder(T_pose_flat).squeeze(-1)
		ref_embedding = self.ref_encoder(Nl_ref).squeeze(-1)

		mel_embedding = self.Norm(mel_embedding)
		pose_embedding = self.Norm(pose_embedding)
		ref_embedding = self.Norm(ref_embedding)

		mel_embedding = mel_embedding.reshape(B, T, 512)
		pose_embedding = pose_embedding.reshape(B, T, 512)
		ref_embedding = ref_embedding.reshape(B, N_l, 512)

		position_v_encoding = self.position_v(pose_embedding)
		position_a_encoding = self.position_a(mel_embedding)

		modality_v = self.modality(torch.ones((B, T), dtype=torch.long))
		modality_a = self.modality(2 * torch.ones((B, T), dtype=torch.long))
		modality_r = self.modality(3 * torch.ones((B, N_l), dtype=torch.long))

		pose_tokens = pose_embedding + position_v_encoding + modality_v
		audio_tokens = mel_embedding + position_a_encoding + modality_a
		ref_tokens = ref_embedding + modality_r

		input_tokens = torch.cat((ref_tokens, audio_tokens, pose_tokens), dim=1)

		output_tokens = self.transformer_encoder(input_tokens)

		lip_embedding = output_tokens[:, N_l:N_l + T, :]
		jaw_embedding = output_tokens[:, N_l + T:, :]
		output_mouse_landmark = self.mouse_keypoint_map(lip_embedding)
		output_jaw_landmark = self.jaw_keypoint_map(jaw_embedding)

		predict_content = torch.cat([output_jaw_landmark, output_mouse_landmark], dim=2)
		predict_content = predict_content.reshape(B, T, 57, 2)
		predict_content = predict_content.reshape(B * T, 57, 2).permute(0, 2, 1)
		return predict_content


class RendererInference(nn.Module):
	def __init__(self, flow_module, translation):
		super().__init__()
		self.flow_module = flow_module
		self.translation = translation

	def forward(self, face_frame_img, target_sketches, ref_N_frame_img, ref_N_frame_sketch, audio_mels):
		wrapped_h1, wrapped_h2, wrapped_ref = self.flow_module(
			ref_N_frame_img, ref_N_frame_sketch, target_sketches
		)

		target_sketches_cat = torch.cat([target_sketches[:, i] for i in range(target_sketches.size(1))], dim=1)
		gt_face = face_frame_img.reshape(-1, 3, face_frame_img.size(3), face_frame_img.size(4))
		gt_mask_face = gt_face.clone()
		gt_mask_face[:, :, gt_mask_face.size(2) // 2:, :] = 0

		translation_input = torch.cat([gt_mask_face, target_sketches_cat], dim=1)
		generated_face = self.translation(translation_input, wrapped_ref, wrapped_h1, wrapped_h2, audio_mels)
		return generated_face, wrapped_ref


def __export_landmark_generator(checkpoint_path, output_path):
	print('Loading landmark generator...')
	model = Landmark_generator(T=5, d_model=512, nlayers=4, nhead=4, dim_feedforward=1024, dropout=0.1)
	model = load_checkpoint(model, checkpoint_path)

	wrapper = LandmarkGeneratorInference(model)
	wrapper.eval()
       
	B, T, Nl = 1, 5, 15
	T_mels = torch.randn(B, T, 1, 80, 16)
	T_pose = torch.randn(B, T, 2, 74)
	Nl_pose = torch.randn(B, Nl, 2, 74)
	Nl_content = torch.randn(B, Nl, 2, 57)

	print('Testing PyTorch forward pass...')
	with torch.no_grad():
		torch_out = wrapper(T_mels, T_pose, Nl_pose, Nl_content)
	print(f'PyTorch output shape: {torch_out.shape}')

	print('Exporting landmark generator to ONNX...')
	with torch.no_grad():
		torch.onnx.export(
			wrapper,
			(T_mels, T_pose, Nl_pose, Nl_content),
			output_path,
			opset_version=16,
			input_names=['T_mels', 'T_pose', 'Nl_pose', 'Nl_content'],
			output_names=['predict_content']
		)
	print('Landmark generator exported to:', output_path)

	import onnxruntime as ort
	session = ort.InferenceSession(output_path)
	onnx_out = session.run(None, {
		'T_mels': T_mels.numpy(),
		'T_pose': T_pose.numpy(),
		'Nl_pose': Nl_pose.numpy(),
		'Nl_content': Nl_content.numpy(),
	})
	torch_out_np = torch_out.detach().numpy()
	diff = np.abs(onnx_out[0] - torch_out_np).max()
	print(f'Landmark generator verification - max diff: {diff:.6f}')
	
def export_landmark_generator(checkpoint_path, output_path):
    import torch
    import numpy as np
    import onnxruntime as ort

    print('Loading landmark generator...')
    model = Landmark_generator(
        T=5,
        d_model=512,
        nlayers=4,
        nhead=4,
        dim_feedforward=1024,
        dropout=0.1
    )
    model = load_checkpoint(model, checkpoint_path)

    wrapper = LandmarkGeneratorInference(model)
    wrapper.eval()

    # ---- Dummy inputs (only for export tracing) ----
    B, T, Nl = 1, 5, 15
    T_mels = torch.randn(B, T, 1, 80, 16)
    T_pose = torch.randn(B, T, 2, 74)
    Nl_pose = torch.randn(B, Nl, 2, 74)
    Nl_content = torch.randn(B, Nl, 2, 57)

    print('Testing PyTorch forward pass...')
    with torch.no_grad():
        torch_out = wrapper(T_mels, T_pose, Nl_pose, Nl_content)
    print(f'PyTorch output shape: {torch_out.shape}')

    print('Exporting landmark generator to ONNX...')
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (T_mels, T_pose, Nl_pose, Nl_content),
            output_path,
            opset_version=16,
            input_names=['T_mels', 'T_pose', 'Nl_pose', 'Nl_content'],
            output_names=['predict_content'],
            dynamic_axes={
              'T_mels': {1: 'T'},            
              'T_pose': {1: 'T'},
              'Nl_pose': {1: 'Nl'},
              'Nl_content': {1: 'Nl'},
              'predict_content': {1: 'T'}
            }
          )

    print('Landmark generator exported to:', output_path)

    

def export_renderer(checkpoint_path, output_path, N_ref=25):
	print('Loading renderer...')
	from models.video_renderer import Renderer
	full_renderer = Renderer()
	full_renderer = load_checkpoint(full_renderer, checkpoint_path)

	wrapper = RendererInference(full_renderer.flow_module, full_renderer.translation)
	wrapper.eval()

	H, W = 256, 256  # Image size
	T_sketch = 5

	# Use the dynamic N_ref provided as argument
	face_frame_img = torch.randn(1, 1, 3, H, W)
	target_sketches = torch.randn(1, T_sketch, 3, H, W)
	ref_N_frame_img = torch.randn(1, N_ref, 3, H, W)
	ref_N_frame_sketch = torch.randn(1, N_ref, 3, H, W)
	audio_mels = torch.randn(1, 1, 1, 80, 16)

	print('Testing PyTorch forward pass...')
	with torch.no_grad():
		torch_out = wrapper(face_frame_img, target_sketches, ref_N_frame_img, ref_N_frame_sketch, audio_mels)
		generated_face, wrapped_ref = torch_out
		print(f'Generated face shape: {generated_face.shape}')
		print(f'Wrapped ref shape: {wrapped_ref.shape}')

		print('Exporting renderer to ONNX...')
		with torch.no_grad():
			torch.onnx.export(
				wrapper,
				(face_frame_img, target_sketches, ref_N_frame_img, ref_N_frame_sketch, audio_mels),
				output_path,
				opset_version=16,
				input_names=['face_frame_img', 'target_sketches', 'ref_N_frame_img', 'ref_N_frame_sketch', 'audio_mels'],
				output_names=['generated_face','wrapped_ref'],
				dynamic_axes={
					'face_frame_img': {2: 'N_face'},
					'ref_N_frame_img': {1: 'N_ref'},
					'ref_N_frame_sketch': {1: 'N_ref'},
					'target_sketches': {1: 'T_sketch'},
					'generated_face': {1: 'T_output'},
					'wrapped_ref': {1: 'N_ref'}
				}
			)
	print('Renderer exported to:', output_path)	


if __name__ == '__main__':
	landmark_ckpt = 'checkpoints/landmark_checkpoint_step000090000.pth'
	renderer_ckpt = 'checkpoints/renderer_checkpoint_step000253500.pth'
	output_dir = 'onnx'

	export_landmark_generator(landmark_ckpt, os.path.join(output_dir, 'ip_lap_landmark_generator.onnx'))
	export_renderer(renderer_ckpt, os.path.join(output_dir, 'ip_lap_renderer.onnx'))
