pip3 install --user --upgrade http://download.pytorch.org/whl/cu75/torch-0.1.10.post2-cp36-cp36m-linux_x86_64.whl 

pip3 install --user --upgrade torchvision

python3 regression.py

ctags-exuberant -R --totals=yes --languages=python

python3 vae.py
