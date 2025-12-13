#!/bin/bash

echo "Starting v1..."
python -u GNN_offline.py > train_gnn_v1.log 2>&1

echo "Starting v2..."
python -u GNN_offline_rot.py > train_gnn_v2.log 2>&1
