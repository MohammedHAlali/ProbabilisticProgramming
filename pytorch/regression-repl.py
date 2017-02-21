import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()

%load_ext autoreload
%autoreload 2

batch_size = 32

random = torch.randn(batch_size)

make_features(random)

random.unsqueeze(1)

X, Y  = get_batch()
fig = plt.figure(1)
# fig.clear()
# ax = fig.gca()
ax.set_title("")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xlim(auto=True)
ax.set_ylim(-33, 10)
ax.scatter( X[:,0].data.numpy(), Y.squeeze().data.numpy() )
ax.legend(loc='upper left')
fig.canvas.draw()
fig.show()


plt.hist(Y.squeeze().data)


c = count(5)


for i in count(1,3):
    print(i)
    if i > 20:
        break

fig = plt.figure(1)
fig.clear()
ax = fig.gca()
ax.set_title("")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xlim(auto=True)
ax.set_ylim(-30, 5)
for i, (x,y) in enumerate(list(zip(x_old, y_old))[1:100:3]):
    print(i)
    x,y = list(zip(*sorted(zip(x,y))))
    # ax.plot( x, y, color='red', alpha=0.1+i/(2*len(x_old)) )
    ax.plot( x, y, color='red', alpha=0.1 )
ax.legend(loc='upper left')
fig.canvas.draw()
fig.show()


range(0,len(x_old),10)

