PROJ_NAME=TPGS
EXP_NAME=custom_no_transient
ROOT_DIR=/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/test_ground/$EXP_NAME
# test if the root dir exists, ask user to remove it if it does
if [ -d "$ROOT_DIR" ]; then
    echo "Directory $ROOT_DIR already exists. Please remove it before running the script."
    read -p "Do you want to remove it now? (y/n) " choice
    if [ "$choice" = "y" ]; then
        rm -rf "$ROOT_DIR"
    else
        exit 1
    fi
fi

PROJ_DIR=$ROOT_DIR/$PROJ_NAME
mkdir -p $ROOT_DIR

cd $ROOT_DIR
git clone git@github.com:ladvu/TPGS.git


cd $PROJ_DIR
COMMIT_HASH=$1
git checkout $COMMIT_HASH

sbatch /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/TPGS/scripts/custom/reconstruct_custom.sh $PROJ_DIR $EXP_NAME