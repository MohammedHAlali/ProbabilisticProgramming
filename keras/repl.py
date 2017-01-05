from keras.layers import Dense , Activation
from keras.models import Sequential
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()

N_ACTIONS=5

model = Sequential()
model.add( Dense( output_dim=N_ACTIONS , input_dim=1 , init="normal" ) )
model.add( Activation( "relu" ) )
model.summary()

z = np.random.randn(200) 
prediction = model.predict( z )

fig = plt.figure(1)
plt.tight_layout()
fig.clear()
ax = fig.gca()
ax.set_title("")
ax.set_xlabel("z")
ax.set_ylabel("ReLU(W*z)")
ax.set_xlim(auto=True)
ax.set_ylim(auto=True)
ax.scatter( z, prediction[:,1], label=''  , color='red'  )
ax.scatter( z, prediction[:,2], label=''  , color='green' )
ax.scatter( z, prediction[:,3], label=''  , color='blue' )
ax.legend(loc='upper left')
fig.canvas.draw()
fig.show()

plt.clf()
plt.hist( prediction )
