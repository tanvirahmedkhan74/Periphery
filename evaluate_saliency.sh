MODEL="data/checkpoints/best_student_model_DUTS25_EC35.pth"
DATASET_DIR="datasets_local"
MODE="multi"

# Unsupervised saliency detection evaluation
for DATASET in ECSSD DUTS-TEST DUT-OMRON
do
    python evaluate.py --eval-type saliency --dataset-eval $DATASET \
            --model-weights $MODEL --evaluation-mode $MODE --apply-bilateral --dataset-dir $DATASET_DIR
done


