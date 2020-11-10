SRC=fr
TGT=en
VALID_SET="IWSLT17.TED.dev2010.fr-en"

echo "Preprocessing training data..."

for LANG in "${SRC}" "${TGT}";
do
    cat "train.tags.${SRC}-${TGT}.${LANG}" \
        | grep -v '<url>' \
        | grep -v '<talkid>' \
        | grep -v '<keywords>' \
        | grep -v '<speaker>' \
        | grep -v '<reviewer' \
        | grep -v '<translator' \
        | grep -v '<doc' \
        | grep -v '</doc>' \
        | sed -e 's/<title>//g' \
        | sed -e 's/<\/title>//g' \
        | sed -e 's/<description>//g' \
        | sed -e 's/<\/description>//g' \
        | sed 's/^\s*//g' \
        | sed 's/\s*$//g' \
        > "train.${SRC}-${TGT}_preprocessed.${LANG}"
done

echo "Preprocessing valid data..."
for LANG in "${SRC}" "${TGT}"; do
    grep '<seg id' "${VALID_SET}.${LANG}.xml" \
        | sed -e 's/<seg id="[0-9]*">\s*//g' \
        | sed -e 's/\s*<\/seg>\s*//g' \
        | sed -e "s/\’/\'/g" \
        > "valid.${SRC}-${TGT}_preprocessed.${LANG}"
done
