# Import libraries
import matplotlib.pyplot as plt
import matplotlib
import numpy as np



# App10 after 10, 20, 30, 40, and 50 runs
estimations = [[] for _ in range(5)]
actuals = [[] for _ in range(5)]


with open("res_experiment_1_epoch_10_app10.csv") as f:
    for i, line in enumerate(f):
        if i == 0:
            continue

        estimation, actual = map(int, line.strip().split(","))
        # we look only at the first 50 points
        if i == 51:
            break

        estimations[(i-1)//10].append(estimation)
        actuals[(i-1)//10].append(actual)


avg_actuals = []
for i in range(5):
    avg_actuals.append(sum(actuals[i]) / 10)


fig, ax = plt.subplots()

ax.set(xlabel='Number of invocations of the application', ylabel='time (ms)',
       title='Runtime estimations over the number of runs')
# Plot a line between the means of each dataset
# testing
# avg_actuals.append(3880)
# avg_actuals.append(4080)
# avg_actuals[2] -= 100
plt.plot(avg_actuals, label='actual average')
# Save the default tick positions, so we can reset them...
plt.boxplot(estimations)

plt.legend(loc="lower right")

# Reset the xtick locations.
plt.xticks([1,2,3,4,5,6], [10,20,30,40,50])


plt.savefig("fig.pdf")
plt.show()
