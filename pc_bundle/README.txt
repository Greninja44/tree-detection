TRAIN ON WINDOWS PC (NVIDIA GPU), DEPLOY ON JETSON
===================================================
1. Unzip tree_datasets.zip INTO THIS FOLDER so you have:
       train_on_pc.py   data/images  data/labels   data_path/images  data_path/labels
2. In PowerShell:
       python -m venv venv ; .\venv\Scripts\Activate.ps1
       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
       pip install -r requirements.txt
       python train_on_pc.py
   (CUDA out-of-memory?  python train_on_pc.py --batch 8)   Roughly 20-40 min total on a mid-range GPU.
3. It prints TEST mAP when done and writes:
       results\tree_trunk.pt   results\forest_path_seg.pt
4. Copy them to the Jetson (replace <jetson-ip>):
       scp results\tree_trunk.pt results\forest_path_seg.pt autogenbot@<jetson-ip>:/home/autogenbot/tree-detection/models/
Datasets: FinnWoodlands (Tampere Univ., (c) Springer - research use) and ForTrunkDet (Zenodo 5213825, CC BY 4.0).
