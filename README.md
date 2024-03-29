# gsa-displacy-demo

![GSA Example Demo](./docs/screenshot-gsa.png)

### Setup
Make sure to have conda installed
- create new conda environment: `conda create --name=gsa_displacy_demo python=3.9`
- activate new conda env: `conda activate gsa_displacy_demo`
- enable shell script to be executed: `chmod +x setup.sh`
- install dependencies using shell script: `bash setup.sh`

## Alternative Setup
Use this when you are unable to `python -m install` the spacy models
- create new conda env: `conda create --name=gsa_displacy_demo python=3.9`
- activate new conda env: `conda activate gsa_displacy_demo`
- pip install requirements: `pip install -r requirements.txt`
- manually download spacy models:
    Download the following models from [link](https://github.com/explosion/spacy-models/releases)
    - en_core_web_sm
    - en_core_web_lg
    - en_core_web_trf
- move model tar files to project subfolder called 'models/', and extract each tar

### Run
to run:
- activate conda environment (if note already activated): `conda activate gsa_displacy_demo`
- start streamlit app: `streamlit run main.py`
