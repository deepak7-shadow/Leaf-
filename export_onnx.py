"""Re-export ONNX model from best_weights.pth with automatic class count detection."""
import sys
import io
import os
import torch
import torchvision.models as models

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

weights_path = 'models/best_weights.pth'
onnx_path = 'models/mobilenet_v3_small.onnx'

if not os.path.exists(weights_path):
    print(f"[ERROR] {weights_path} not found.")
    sys.exit(1)

print(f'Loading {weights_path}...')
checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)

if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

# Auto-detect number of classes from classifier weight shape
out_features = state_dict['classifier.3.weight'].shape[0]
print(f'Detected {out_features} output classes from weights.')

model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, out_features)
model.load_state_dict(state_dict)
model.eval()

dummy = torch.randn(1, 3, 224, 224)

print('Exporting ONNX...')
torch.onnx.export(
    model,
    dummy,
    onnx_path,
    opset_version=17,
    dynamo=False,
    export_params=True,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
)

size_mb = os.path.getsize(onnx_path) / 1024 / 1024
print(f'ONNX size: {size_mb:.2f} MB')
print(f'SUCCESS - exported to {onnx_path} ({out_features} classes)')
