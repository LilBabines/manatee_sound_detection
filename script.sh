#!/bin/bash

source ~/python-venv/lamantin/bin/activate


for cfg in dasheng wav2vec ; do

  for ds in train_val_0.02 train_val_0.05 train_val_0.1 train_val_0.25 training_set_all_deep_corrected; do
    python main.py --config-name=$cfg data.path_csv=data/training_set/${ds}.csv hydra.run.dir=outputs/mixup_06/${cfg}/${ds}
  done
done