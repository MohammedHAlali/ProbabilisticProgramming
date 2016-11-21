
ROOT="http://yann.lecun.com/exdb/mnist/"

wget $ROOT'train-labels-idx1-ubyte.gz'

wget $ROOT'train-images-idx3-ubyte.gz'

wget $ROOT't10k-labels-idx1-ubyte.gz'

wget $ROOT't10k-images-idx3-ubyte.gz'

pip install --user --upgrade graphviz

cd /usr
myfind zathura

pip search opencv

pip install --user --upgrade opencv-python

cd /home/abergman/.julia/v0.5/MXNet/examples/mnist/mlp.jl
