
FOLDER2CHECK=$1

for file in $FOLDER2CHECK/*; do 
    printf "$(basename $file) " 
    ls $file | wc -l
done