"""Quick confidence verification script — run from Leaf/ project root."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else "."))

from core.pipeline import AINozzlePipeline
from utils.synthetic_generator import SyntheticSceneGenerator

gen = SyntheticSceneGenerator()
p = AINozzlePipeline()

tests = [
    ("REAL leaf only",          dict(include_real=True,  include_fake=False)),
    ("FAKE leaf only",          dict(include_real=False, include_fake=True)),
    ("BOTH leaves in same frame", dict(include_real=True, include_fake=True)),
]

for label, kwargs in tests:
    print(f"\n--- {label} ---")
    scene, _ = gen.generate_scene(**kwargs)
    res = p.process_frame(scene)
    if not res.detections:
        print("  No detections (Stage 1 found nothing)")
    for d in res.detections:
        spray_txt = "[SPRAY]" if d["should_spray"] else "[NO SPRAY]"
        print(f"  {d['class_name']:30s}  Conf: {d['confidence']*100:5.1f}%  {spray_txt}")
