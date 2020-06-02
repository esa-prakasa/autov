import matplotlib.pyplot as plt

ax = []
for i in range(2):
	ax.append("")

fig, (ax[0:1]) = plt.subplots(1, 2)
fig.suptitle('Horizontally stacked subplots')

x = []
y =[]
y2 = []
for i in range(100):
	x.append(i)
	yv = x[i] * x[i]
	y.append(yv)
	y2.append(1000-y[i])


# summarize history for accuracy
ax[0].plot(x)
ax[0].plot(y)
ax[0].set_title("model accuracy")
ax[0].set_ylabel("accuracy")
ax[0].set_xlabel('epoch')
ax[0].legend(['train', 'test'], loc='upper left')

ax[1].plot(x)
ax[1].plot(y2)
ax[1].set_title("model accuracy")

plt.show()