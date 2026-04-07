

cd dependencies/sam1

pip install .

cd ../sam2 

pip install .

cd ..

cd RAFT/

# bash download_models.sh
python download_models.py

cd ../../scripts/checkpoints

bash download_ckpts.sh



