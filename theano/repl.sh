pip3 install --user --upgrade Theano

pip3 install --user --upgrade psutil

pip3 install --user --upgrade memory_profiler

pip search memory_profiler

python3 -m memory_profiler ./mem-prof.py

export OMP_NUM_THREADS=1
python3 ~/.local/lib/python3.5/site-packages/theano/misc/check_blas.py -q -B 2000
# 4.714

export OMP_NUM_THREADS=2
python3 ~/.local/lib/python3.5/site-packages/theano/misc/check_blas.py -q -B 2000
# 4.68

python3 ~/.local/lib/python3.5/site-packages/theano/misc/check_blas.py 
#226.7
#74.49 after custom atlas install

export OMP_NUM_THREADS=4
python ~/.local/lib/python3.5/site-packages/theano/misc/elemwise_openmp_speedup.py
# Timed with vector of 200000 elements
# Fast op time without openmp 0.000182s with openmp 0.000115s speedup 1.58
# Slow op time without openmp 0.006456s with openmp 0.001645s speedup 3.93

# https://www.reddit.com/r/MachineLearning/comments/4ekywt/tensorflow_vs_theano_which_to_learn/d21xhin/?st=ixm8s60u&sh=7d978275
export THEANO_FLAGS="device=cpu,floatX=float32,optimizer=None,compute_test_value=raise"
time python3 ./updates_test.py

export THEANO_FLAGS=""
time python3 ./updates_test.py

