#!/bin/bash
set -euo pipefail

CODE=(
	"cse493g1/transformer_layers.py"
	"cse493g1/simclr/contrastive_loss.py"
	"cse493g1/simclr/data_utils.py"
	"cse493g1/simclr/utils.py"
    "cse493g1/rlhf/image_captioning_rlhf.py"
)
NOTEBOOKS=(
	"Transformer_Captioning.ipynb"
	"Self_Supervised_Learning.ipynb"
    "RLHF_Image_Captioning.ipynb"
)

FILES=( "${CODE[@]}" "${NOTEBOOKS[@]}" )
ZIP_FILENAME="a5_code_submission.zip"

C_R="\e[31m"
C_G="\e[32m"
C_BLD="\e[1m"
C_E="\e[0m"

for FILE in "${FILES[@]}"
do
	if [ ! -f ${FILE} ]; then
		echo -e "${C_R}Required file ${FILE} not found, Exiting.${C_E}"
		exit 0
	fi
done

echo -e "### Zipping file ###"
rm -f ${ZIP_FILENAME}
zip -q "${ZIP_FILENAME}" -r ${NOTEBOOKS[@]} $(find . -name "*.py") $(find . -name "*.pyx") -x "makepdf.py"

echo -e "### Done! Please submit ${ZIP_FILENAME} to Gradescope. ###"
