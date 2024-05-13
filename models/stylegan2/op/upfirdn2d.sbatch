#!/bin/env bash
#SBATCH --cpus-per-task=1       # Количество ядер CPU, выделенных для одного процесса
#SBATCH --gpus=1                # Количество используемых GPU
#SBATCH --time=10:00            # Предельное количество времени на задачу

module load Python gnu8
source activate editing
cd /home/mdnikolaev/ImageEditing/Barbershop/models/stylegan2/op
python upfirdn2d.py
echo "Done"