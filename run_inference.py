import sys
import os
import subprocess

# 1. Resolve path issues
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 2. Change to opv2v directory
os.chdir(os.path.join(current_dir, 'opv2v'))

print("="*50)
print("Step 1: Running dynamic BEV segmentation inference...")
print("="*50)
subprocess.run([
    sys.executable, 
    'opencood/tools/inference_camera.py', 
    '--model_dir', 'opencood/logs/cobevt'
], check=True)

print("\n" + "="*50)
print("Step 2: Running static BEV segmentation inference...")
print("="*50)
subprocess.run([
    sys.executable, 
    'opencood/tools/inference_camera.py', 
    '--model_dir', 'opencood/logs/cobevt_static',
    '--model_type', 'static'
], check=True)

print("\n" + "="*50)
print("Step 3: Merging dynamic and static results...")
print("="*50)
subprocess.run([
    sys.executable, 
    'opencood/tools/merge_dynamic_static.py', 
    '--dynamic_path', 'opencood/logs/cobevt',
    '--static_path', 'opencood/logs/cobevt_static',
    '--output_path', 'merge_results'
], check=True)

print("\n" + "="*50)
print("All inference completed! Results are in opv2v/merge_results folder!")
print("="*50)
