MODEL="data/checkpoints/best_student_model_DUTS50.pth"
DATASET_DIR="datasets_local"

# Single object discovery evaluation
for DATASET in VOC07 VOC12 COCO20k
do
    python evaluate.py --eval-type uod --dataset-eval $DATASET \
            --model-weights $MODEL --evaluation-mode single --dataset-dir $DATASET_DIR
done


