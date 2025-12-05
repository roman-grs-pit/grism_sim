#!/bin/bash

if [ -z "$1" ] # check if argument was provided
then
  echo No target directory provided. Please provide a directory containing fits files.
  exit
fi

for fn in "$1"/grism_*_detSCA??.fits;
do

  fn=$(basename "${fn}")
  fn_stripped=${fn%.*} # strip .fits from filename

  det_num=${fn_stripped#*SCA} # strip anything preceding the detector number
  det_num=${det_num%[._]*} # strip anything after the detector number

  ra=${fn_stripped#*ra} # extract ra from filename
  ra=${ra%%[^0-9.]*}

  dec=${fn_stripped#*dec} # extract ra from filename
  dec=${dec%%[^0-9.]*}

  echo Processing "$fn" for detector SCA"$det_num"

  det_num=${det_num#0} # strip leading zeros if single digit detector number

  romanisim-make-image  --extra-counts "$fn" 5 \
  --radec "$ra" "$dec" --date 2026-10-01T12:00:00.000 \
  --sca "$det_num" --level 1 \
  --pretend-spectral True --nobj 0 \
  --bandpass F158 \
  "${fn_stripped}_l1.asdf"

  romanisim-make-image  --extra-counts "$fn" 5 \
  --radec "$ra" "$dec" --date 2026-10-01T12:00:00.000 \
  --sca "$det_num" --level 2 \
  --pretend-spectral True --nobj 0 \
  --bandpass F158 \
  "${fn_stripped}_l2.asdf"

done
