import matplotlib.pyplot as plt
import pandas as pd


# plt_graph() function is used to plot the graph of the scores, average scores and the solved requirement
def plt_graph(episodes, scores, avg_scores, goals, env_name, model_name, exp_name, save_path='./'):
    df = pd.DataFrame({'x': episodes, 'Score': scores, 'Average Score': avg_scores, 'Solved Requirement': goals})

    plt.plot('x', 'Score', data=df, marker='', color='blue', linewidth=2, label='Score')
    plt.plot('x', 'Average Score', data=df, marker='', color='orange', linewidth=2, linestyle='dashed',
             label='AverageScore')
    plt.plot('x', 'Solved Requirement', data=df, marker='', color='red', linewidth=2, linestyle='dashed',
             label='Solved Requirement')
    plt.legend()
    # 限制y轴范围
    plt.ylim(70, 110)
    plt.savefig(save_path + '/' + '{}_{}_{}.png'.format(env_name, model_name, exp_name))
    plt.close()
