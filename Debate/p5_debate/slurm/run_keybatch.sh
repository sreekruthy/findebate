#!/bin/bash
#SBATCH --job-name=findebate_p5_kb
#SBATCH --partition=cpu_student
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-5%1
#SBATCH --output=/dgxa_home/se23ucse176/findebate/logs/p5/kb_%A_%a.out
#SBATCH --error=/dgxa_home/se23ucse176/findebate/logs/p5/kb_%A_%a.err

source /dgxa_home/se23ucse176/findebate/venv/bin/activate

PROJECT_DIR=/dgxa_home/se23ucse176/findebate/p5_debate
FILE_LIST=$PROJECT_DIR/slurm/${BATCH_FILE}
SOURCE_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $FILE_LIST)

echo "=============================="
echo "Task  : $SLURM_ARRAY_TASK_ID"
echo "File  : $SOURCE_FILE"
echo "=============================="

cd $PROJECT_DIR
python run_debate.py \
    --source_file "$SOURCE_FILE" \
    --p4_dir  "/dgxa_home/se23ucse176/findebate/outputs" \
    --p3_dir  "/dgxa_home/se23ucse176/findebate/p3_outputs" \
    --out_dir "/dgxa_home/se23ucse176/findebate/p5_outputs" \
    --log_dir "/dgxa_home/se23ucse176/findebate/logs/p5"

EXIT_CODE=$?
echo "Finished $SOURCE_FILE with exit code $EXIT_CODE"
exit $EXIT_CODE
