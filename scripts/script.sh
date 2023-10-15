#!/bin/bash
rsync -auhP --info=progress2 --no-inc-recursive chawin/transfer-defense/results/cifar10_robustbench-wang23-wrn-70-16 savio:/global/home/users/huang33176/scratch/transfer-defense/results  
