# Push-T DDIM training run

- Status: stopped early at the user's epoch-150 cutoff
- Last fully completed epoch: 149 (150 completed epochs, numbered 0-149)
- Stop time: 2026-09-01 22:50 PDT (2026-09-02T05:50Z)
- Lambda instance: `6a0fba17eb32419ea55bc4698686e288`
- Lambda GPU: NVIDIA H100 80GB HBM3
- Remote training commit: `1e208519b157234d436b9acf07b565d3ba432420`
- Training command: `.venv/bin/python -u pusht_ddim.py train --device cuda --epochs 200 --batch-size 64 --learning-rate 2e-4 --timesteps 1000 --checkpoint checkpoints/pusht_ddim-200-epochs.pt --seed 42`
- Checkpoint SHA-256: `f3de41d369b43ba5000af86e4aadbe9f7b81fc3ce89aa930ee7454bea40d2c11`
- Sample archive SHA-256: `f994ba3ed9211121526884e459d3ae8bd99acdbc56accd2db8346b09ad9255f0`
- Verified output files: 1,921 nonempty files; 0 empty files
- Initial losses: train `0.42132471`, validation `0.23991354`
- Final losses: train `0.02283495`, validation `0.01769728`
- Best training loss: `0.02102893` at epoch 146
- Best validation loss: `0.01769728` at epoch 149
- Last-10 mean losses: train `0.02243948`, validation `0.02825534`

The checkpoint was saved after epoch 149. Epoch 150 had begun but was interrupted, so its partial optimizer updates are not present in the checkpoint. The epoch-150 sample visualizations were generated immediately after epoch 149 and are included.
