FOLDER2CHECK=$1
N=$2
ls $FOLDER2CHECK | head -n $N | tail -n 1  # Get 5th line