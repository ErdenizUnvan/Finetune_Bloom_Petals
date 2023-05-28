# Finetune_Bloom_Petals
Use Ubuntu
If you have windows, then use wsl for ubuntu.
#At ubuntu terminal
#Install pipenv
pip install pipenv
#Create virtual environment
pipenv --three
#Activate virtual environment
pipenv shell
#At activate virtual environment
pip install torch==2.0.0
pip install petals==1.1.3
pip install transformers==4.25.1
pip install wandb
#Sign in to wandb then login to wandb
#https://wandb.ai/
python3 bloom.py
python3 bloom2.py
python3 bloom3.py