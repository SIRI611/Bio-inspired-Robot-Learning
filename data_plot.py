import matplotlib.pyplot as plt
import dill
import seaborn as sns
sns.set_style("darkgrid")

def calculate_average_return(episode_return, num):
    episode_return_average = list()
    for index, reward in enumerate(episode_return):
        if index <= num-1:
            average_return = sum(episode_return[:index+1]) / index+1
            episode_return_average.append(average_return)
        else:
            average_return = sum(episode_return[index-num+1:index+1]) / num
            episode_return_average.append(average_return)
    return episode_return_average



with open("results/episode_return.pkl", "rb") as f:
    episode_return = dill.load(f)


plt.figure(0)
episode_return_average = calculate_average_return(episode_return, 10)
plt.plot(range(len(episode_return)), episode_return)
plt.plot(range(len(episode_return)), episode_return_average)
plt.show()