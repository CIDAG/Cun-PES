#!/bin/sh

# check number of args
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 mode settings_file"
    echo "Modes: lm_only tsne_forward everything"
    exit 1
fi

# check if mode is valid
if [ ! "$1" = "lm_only" ] && \
   [ ! "$1" = "tsne_forward" ] && \
   [ ! "$1" = "everything" ]; then
    echo "Invalid mode."
    exit 1
fi

# get project absolute path
prg=$0
if [ ! -e "$prg" ]; then
    case $prg in
        (*/*) exit 1;;
        (*) prg=$(command -v -- "$prg") || exit;;
    esac
fi
dir=$(
    cd -P -- "$(dirname -- "$prg")" && pwd -P
) || exit
prg=$dir/$(basename -- "$prg") || exit

actual_dir=$(dirname $prg)

# get settings file absolute path
settings_ini="$(cd "$(dirname "$2")"; pwd)/$(basename "$2")"

# create outputs folder if doesn't exist
grep "outputs_folder" $settings_ini | awk '{print $3}' | xargs mkdir -p

# ignore first step of the program?
if [ ! "$1" = "tsne_forward" ]; then
    python3 ${actual_dir}/00_lmkmeans.py ${settings_ini}
fi

# run only the first step of the program?
if [ "$1" = "lm_only" ]; then
    exit 0
fi

# run everything else
python3 ${actual_dir}/01_tsne.py ${settings_ini}
python3 ${actual_dir}/02_treat_tsne.py ${settings_ini}

echo "Done!"
