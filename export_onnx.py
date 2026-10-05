"""Re-export ONNX model from best_weights.pth using dynamo=False."""
import sys
import io
import torch
import torchvision.models as models
import os

# Force UTF-8 output to avoid emoji encoding errors on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print('Loading best_weights.pth...')
checkpoint = torch.load('models/best_weights.pth', map_location='cpu', weights_only=True)

model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)

if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint: epoch={checkpoint.get('epoch','?')}")
elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
    print('Loaded from state_dict key')
else:
    model.load_state_dict(checkpoint)
    print('Loaded raw state dict')

model.eval()
dummy = torch.randn(1, 3, 224, 224)

print('Exporting ONNX with dynamo=False ...')
torch.onnx.export(
    model,
    dummy,
    'models/mobilenet_v3_small.onnx',
    opset_version=17,
    dynamo=False,
    export_params=True,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
)

size_mb = os.path.getsize('models/mobilenet_v3_small.onnx') / 1024 / 1024
print(f'ONNX size: {size_mb:.2f} MB')
print('SUCCESS - model ready!')
