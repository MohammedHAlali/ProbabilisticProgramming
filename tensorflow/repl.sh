pip3 install --upgrade --user neovim

pip install --upgrade --user seaborn

pip install --upgrade --user tensorflow
# tensorflow-0.12.1-cp27-cp27mu-manylinux1_x86_64.whl

pip3 install --upgrade --user tensorflow

pip3 install  --user https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp35-cp35m-linux_x86_64.whl

pip install --user --upgrade tfdebugger

tensorboard --logdir=. --verbose --reload_interval=1

wget https://raw.githubusercontent.com/ericjang/tdb/master/notebooks/mnist_demo.ipynb

wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py

python ./mnist_with_summaries.py
tensorboard --logdir=/tmp/tensorflow --verbose --reload_interval=1

rm -rf ./tf_logs/*
