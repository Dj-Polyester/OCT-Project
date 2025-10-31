FOLDER2CHECK=$1
count=0
for file in $FOLDER2CHECK/*; do 
    printf "$(basename $file) " 
    files=$(ls $file | wc -l)
    echo $files
    ((count += files))
done
echo "Total files: $count"