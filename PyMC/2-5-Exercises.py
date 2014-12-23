#

fig = plt.figure()
fig.clear()
ax = fig.gca()
ax.set_title("Why?")
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$\beta$")
ax.set_xlim(auto=True)
ax.set_ylim(auto=True)
ax.scatter( alpha_samples, beta_samples
        , alpha = 0.1
        )
fig.canvas.draw()
fig.show()
