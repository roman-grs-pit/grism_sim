#!/bin/bash

if [ -z "$1" ] # check if argument was provided
then
  echo No target directory provided. Please provide a directory containing fits files.
  exit
fi

for fn in "$1"/grism_*_detSCA??.fits;
do

  fn_stripped=$(basename "${fn}")
  fn_stripped=${fn_stripped%.*} # strip .fits from filename

  det_num=${fn_stripped#*SCA} # strip anything preceding the detector number
  det_num=${det_num%[._]*} # strip anything after the detector number

  ra=${fn_stripped#*ra} # extract ra from filename
  ra=${ra%%[^0-9.]*}

  dec=${fn_stripped#*dec} # extract ra from filename
  dec=${dec%%[^0-9.]*}

  pa=${fn_stripped#*pa} # extract pa from filename
  pa=${pa%%[^0-9.]*}

  printf %"$(tput cols)"s | tr " " -
  echo Processing "$fn" for detector SCA"$det_num"

  det_num=${det_num#0} # strip leading zeros if single digit detector number
  l2fn="${fn_stripped}_l2.asdf"

  echo Generating "$l2fn"

  romanisim-make-image \
  --extra-counts "$fn" 5 \
  --radec "$ra" "$dec" --roll "$pa" \
  --date 2026-01-01T12:00:00.000 \
  --ma_table_number 1036 \
  --bandpass GRISM \
  --sca "$det_num" \
  --rng_seed 42 \
  --usecrds \
  --nobj 0 \
  --stpsf \
  --level 2 \
  l2fn

done


