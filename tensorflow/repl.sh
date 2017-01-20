pip3 install --upgrade --user neovim

pip install --upgrade --user seaborn

pip install --upgrade --user tensorflow
# tensorflow-0.12.1-cp27-cp27mu-manylinux1_x86_64.whl

pip3 install --upgrade --user tensorflow

pip3 install  --user https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp35-cp35m-linux_x86_64.whl

pip install --user --upgrade tfdebugger

wget https://raw.githubusercontent.com/ericjang/tdb/master/notebooks/mnist_demo.ipynb

wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py

python ./mnist_with_summaries.py
tensorboard --logdir=/tmp/tensorflow --verbose --reload_interval=1

rm -rf ./tf_logs/*

tensorboard --logdir=. --verbose --reload_interval=1

virtualenv-3.5 --verbose venv

source venv/bin/activate

pip install ipython tensorflow matplotlib pandas sklearn scipy

rm -rf ./venv

wget https://gist.githubusercontent.com/ilblackdragon/dfaefa3ac6097dbc27de/raw/31934e210685f590b883c3afb9ea70ca8bd3df29/titanic_categorical1.py

wget https://raw.githubusercontent.com/ilblackdragon/tf_examples/master/data/titanic_train.csv

ctags-exuberant -R --totals=yes --languages=python venv/

git clone https://github.com/tensorflow/models.git

