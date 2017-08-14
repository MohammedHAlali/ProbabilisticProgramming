cd apps/

git clone https://github.com/eBay/bayesian-belief-networks.git

cd bayesian-belief-networks

python setup.py install --user

pip install --user -r requirements.txt

pip install --user sphinx

cd docs/tutorial

make html

cd _build/html

python -m SimpleHTTPServer 8000
