import torch
from src.models.modnet import MODNet

# Load your PyTorch model
model = MODNet(backbone_pretrained=False)
checkpoint = torch.load('modnet_photographic_portrait_matting.ckpt', map_location='cpu')
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict, strict=False)
model.eval()

# Create a dummy input
dummy_input = torch.randn(1, 3, 512, 512)

# Export the model
torch.onnx.export(model, dummy_input, "modnet.onnx", 
                  opset_version=11, 
                  input_names=['input'], 
                  output_names=['output'], 
                  dynamic_axes={'input': {0: 'batch_size'}, 
                                'output': {0: 'batch_size'}})

print("ONNX model has been saved as modnet.onnx")
