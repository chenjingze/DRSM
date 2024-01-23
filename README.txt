

Train:

PYTHONPATH='.' python drsm/main.py --config-path drsm/configs/icassp/Drsm/full_5000_boxs.py

Recontruct_pointclouds:

python recontruct_pointclouds.py --root_path ./logs/icassp/marionette/full_5000_marionette_tloss_dy/estm --out_postfix lp_recon --depth_smoother


