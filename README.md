# scripts and helper utilities to simulate Roman grism images

### Instructions for setting up NERSC environment needed to run Roman grism simulations
Given NERSC install, hopefully the following is all that is needed for anyone in the roman group 

```
module load conda

conda activate /global/common/software/m4943/grizli0

PYTHONPATH=$PYTHONPATH:/global/common/software/m4943/grizli0/grism_sim/py

export STPSF_PATH="/global/cfs/cdirs/m4943/grismsim/stpsf-data"

export github_dir=/global/common/software/m4943/grizli0/
```

### Full NERSC install (for reference only, most users will need the above option)

```
module load conda

conda create --prefix /global/common/software/m4943/grizli0 python=3.9

conda activate /global/common/software/m4943/grizli0

conda install numpy scipy astropy

cd /global/common/software/m4943/grizli0

git clone https://github.com/roman-grs-pit/grizli.git

cd grizli

pip install -e .
```

Following error:
"ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
seaborn 0.12.1 requires pandas>=0.25, which is not installed.
basemap 1.3.2 requires matplotlib<3.6,>=1.5; python_version >= "3.5", but you have matplotlib 3.9.4 which is incompatible.
basemap 1.3.2 requires numpy<1.23,>=1.21; python_version >= "3.7", but you have numpy 1.26.4 which is incompatible."

```
pip install ".[test]"

pip install stpsf

mkdir /global/cfs/cdirs/m4943/grismsim

export STPSF_PATH="/global/cfs/cdirs/m4943/grismsim/stpsf-data"

cd $STPSF_PATH

cd .. 
#otherwise, get double stpsf-data directory

#check for updated file here https://stpsf.readthedocs.io/en/latest/installation.html
wget https://stsci.box.com/shared/static/kqfolg2bfzqc4mjkgmujo06d3iaymahv.gz

mv kqfolg2bfzqc4mjkgmujo06d3iaymahv.gz  stpsf-data-LATEST.tar.gz

tar xzvf stpsf-data-LATEST.tar.gz

cd /global/common/software/m4943/grizli0

git clone https://github.com/roman-grs-pit/grism_sim.git

git clone https://github.com/roman-grs-pit/star_fields.git

conda install pysynphot (other machines might need pip install pysynphot instead)

PYTHONPATH=$PYTHONPATH:/global/common/software/m4943/grizli0/grism_sim/py
```
