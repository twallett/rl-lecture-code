#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def violinplot_environment(data, arm_means = None):
    legend_labels = []
    if arm_means == None:
        plt.violinplot(data, showmeans=True)
        plt.xticks([i + 1 for i in range(len(data[0,:]))])
        legend_labels.append(f"Distribution from 0 to {len(data)}")
        plt.xlabel("Bandits (Possible actions)")
        plt.ylabel("Rewards distribution")
        plt.title(f"MultiArmBandits distribution of {len(data[0,:])} bandits")
        plt.legend(labels = legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
    else:
        asplit = int(len(data) / len(arm_means))
        legend_handles = []
        positions = [(i + 1) * len(data[0,:]) for i in range(len(data[0,:]))]
        for j in range(0, len(data[0,:])):
            sep = np.linspace(positions[j]- (0.15 * len(arm_means)), positions[j]+(0.15 * len(arm_means)), len(arm_means))
            for i in range(len(arm_means)):
                xmin = i * asplit
                xmax = (i + 1) * asplit
                color = f'C{i}'
                stats = plt.violinplot(data[xmin:xmax, j], showmeans= True, positions=sep[i:i+1])
                stats['bodies'][0].set_color(color)
                stats['cmeans'].set_color(color)
                stats['cbars'].set_color(color)
                stats['cmins'].set_color(color)
                stats['cmaxes'].set_color(color)
                if j == 0:
                    legend_labels.append(f"Distribution from {xmin} to {xmax}")
                    legend_handles.append(Rectangle((0, 0), 1, 1, color=color, alpha = 0.3))
        plt.xticks(positions, [i + 1 for i in range(len(data[0,:]))])
        plt.xlabel("Bandits (Possible actions)")
        plt.ylabel("Rewards distribution")
        plt.title(f"MultiArmBandits distribution of {len(data[0,:])} bandits")
        plt.legend(handles = legend_handles, labels = legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.savefig('1-1MAB-Violinplot.pdf', bbox_inches='tight')
        plt.show()

def data_average_plot(data, arm_means, top = None):
    if top == None:
        pass
    else:
        arm_means = np.array(arm_means)
        means = np.mean(arm_means, axis = 0)
        indxs = np.linspace(0, len(data[0,:]) - 1, len(data[0,:]))
        for _ in range(len(means) - top):
            indx = means.argmin()
            data = np.delete(data, indx, axis = 1)
            arm_means = np.delete(arm_means, indx, axis = 1)
            indxs = np.delete(indxs, indx)
            means = np.delete(means, indx)
        alist = []
        for row in arm_means:
            alist.append(row)
        arm_means = alist
        
    if len(arm_means) == 1:
        x = np.ones((len(data), len(data[0,:])))
        x = arm_means[0] * x
        for j in range(0, len(data[0,:])):
            if top == None:
                plt.plot([np.cumsum(data[:,j])[i]/(i+1) for i in range(len(data))], color=f"C{j}", label=f"Arm {j+1}")
                plt.plot(x[:,j], linestyle="--")
            else:
                plt.plot([np.cumsum(data[:,j])[i]/(i+1) for i in range(len(data))], color=f"C{j}", label=f"Arm {int(indxs[j] + 1)}")
                plt.plot(x[:,j], linestyle="--") 
    else:
        x = np.ones((len(data), len(data[0,:])))
        asplit = int(len(data)/len(arm_means))
        for i in range(len(arm_means)):
            x[i * asplit : (i+1) * asplit,:] *=  arm_means[i]
        plt.plot(x, linestyle="--")
        if top == None:
            for j in range(0, len(data[0,:])):
                plt.plot([np.cumsum(data[:,j])[i]/(i+1) for i in range(len(data))], color=f"C{j}", label=f"Arm {j+1}")
        else:
            for j in range(0, len(data[0,:])):
                plt.plot([np.cumsum(data[:,j])[i]/(i+1) for i in range(len(data))], color=f"C{j}", label=f"Arm {int(indxs[j] + 1)}")
    plt.xlabel("Steps")
    plt.ylabel("Average of rewards")
    plt.title("MultiArmBandits rolling average of arms data")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.savefig('1-1MAB-DataAvg.pdf', bbox_inches='tight')
    plt.show()

def data_cumulative_plot(data, arm_means, top = None):
    if top == None:
        pass
    else:
        arm_means = np.array(arm_means)
        means = np.mean(arm_means, axis = 0)
        indxs = np.linspace(0, len(data[0,:]) - 1, len(data[0,:]))
        for _ in range(len(means) - top):
            indx = means.argmin()
            data = np.delete(data, indx, axis = 1)
            arm_means = np.delete(arm_means, indx, axis = 1)
            indxs = np.delete(indxs, indx)
            means = np.delete(means, indx)
        alist = []
        for row in arm_means:
            alist.append(row)
        arm_means = alist

    if len(arm_means) == 1:
        x = np.column_stack((np.linspace(0, len(data), len(data) * 10) for _ in range(len(data[0,:]))))
        y = x * arm_means[0]
        for j in range(len(data[0,:])):  # Loop through batches
            if top == None:
                plt.plot(np.cumsum(data[:,j]), color=f"C{j}", label=f"Arm {j+1}")
                plt.plot(x, y[:,j], color = f"C{j}", linestyle = "--")
            else:
                plt.plot(np.cumsum(data[:,j]), color=f"C{j}", label=f"Arm {int(indxs[j] + 1)}")
                plt.plot(x, y[:,j], color = f"C{j}", linestyle = "--")
    else:
        for j in range(len(data[0,:])):  # Loop through batches
            plt.plot(np.cumsum(data[:,j]), color=f"C{j}", label=f"Arm {j+1}")
        asplit = int(len(data) / len(arm_means))
        previous = []
        maxes = []
        for j in range(len(arm_means)):
            xmin = j * asplit
            xmax = (j + 1) * asplit
            x = np.column_stack((np.linspace(xmin, xmax, len(data) * 10) for _ in range(len(data[0,:]))))
            if j == 0:
                y = arm_means[j] * x
                maxes.append(xmax)
                previous.append(y[-1])
            else:
                y = (arm_means[j] * (x-maxes[j-1])) + previous[j-1]
                previous.append(y[-1])
            for _ in range(len(data[0,:])):
                plt.plot(x, y[:,_], color = f"C{_}", linestyle = "--")
    plt.xlabel("Steps")
    plt.ylabel("Cumulative sum of rewards")
    plt.title("MultiArmBandits cumulative sum of arms data")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.savefig('1-1MAB-DataCum.pdf', bbox_inches='tight')
    plt.show()

def model_average_plot(data, rewards, matrix, arm_means, top = None):
    if top == None:
        pass
    else:
        arm_means = np.array(arm_means)
        means = np.mean(arm_means, axis = 0)
        indxs = np.linspace(0, len(data[0,:]) - 1, len(data[0,:]))
        for _ in range(len(means) - top):
            indx = means.argmin()
            data = np.delete(data, indx, axis = 1)
            arm_means = np.delete(arm_means, indx, axis = 1)
            matrix = np.delete(matrix, indx, axis = 1)
            indxs = np.delete(indxs, indx)
            means = np.delete(means, indx)
        alist = []
        for row in arm_means:
            alist.append(row)
        arm_means = alist
    
    if len(arm_means) == 1:
        x = np.ones((len(data), len(data[0,:])))
        x = arm_means[0] * x
        plt.plot([np.cumsum(rewards)[i]/(i+1) for i in range(len(rewards))], label = f'Rewards', color ="Black")
        for j in range(0,len(data[0,:])):
            if top == None:
                plt.plot([np.cumsum(matrix[:,j])[i]/(i+1) for i in range(len(matrix))], color=f"C{j}", label=f"Arm {j+1}")
                plt.plot(x[:,j], linestyle="--")
            else:
                plt.plot([np.cumsum(matrix[:,j])[i]/(i+1) for i in range(len(matrix))], color=f"C{j}", label=f"Arm {int(indxs[j] + 1)}")
                plt.plot(x[:,j], linestyle="--") 
    else:
        x = np.ones((len(data), len(data[0,:])))
        asplit = int(len(data)/len(arm_means))
        for i in range(len(arm_means)):
            x[i * asplit : (i+1) * asplit,:] *=  arm_means[i]
        plt.plot(x, linestyle="--")
        plt.plot([np.cumsum(rewards)[i]/(i+1) for i in range(len(rewards))], label = f'Rewards', color ="Black")
        if top == None:
            for j in range(0, len(matrix[0,:])):
                plt.plot([np.cumsum(matrix[:,j])[i]/(i+1) for i in range(len(matrix))], color=f"C{j}", label=f"Arm {j+1}")
        else:
            for j in range(0, len(matrix[0,:])):
                plt.plot([np.cumsum(matrix[:,j])[i]/(i+1) for i in range(len(matrix))], color=f"C{j}", label=f"Arm {int(indxs[j] + 1)}")
    plt.xlabel("Steps")
    plt.ylabel("Average of rewards")
    plt.title("MultiArmBandits rolling average of arms model")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.savefig('1-1MAB-ModelAvg.pdf', bbox_inches='tight')
    plt.show()

def model_cumulative_plot(data, rewards, matrix, arm_means, top = None):
    if top == None:
        pass
    else:
        arm_means = np.array(arm_means)
        means = np.mean(arm_means, axis = 0)
        indxs = np.linspace(0, len(data[0,:]) - 1, len(data[0,:]))
        for _ in range(len(means) - top):
            indx = means.argmin()
            data = np.delete(data, indx, axis = 1)
            arm_means = np.delete(arm_means, indx, axis = 1)
            matrix = np.delete(matrix, indx, axis = 1)
            indxs = np.delete(indxs, indx)
            means = np.delete(means, indx)
        alist = []
        for row in arm_means:
            alist.append(row)
        arm_means = alist
    
    if len(arm_means) == 1:
        plt.plot(np.cumsum(rewards), label = f'Rewards', color ="Black")
        x = np.column_stack((np.linspace(0, len(data), len(data) * 10) for _ in range(len(data[0,:]))))
        y = x * arm_means[0]
        for j in range(len(matrix[0,:])):  # Loop through batches
            if top == None:
                plt.plot(np.cumsum(matrix[:,j]), color=f"C{j}", label=f"Arm {j+1}")
                plt.plot(x, y[:,j], color = f"C{j}", linestyle = "--")
            else:
                plt.plot(np.cumsum(matrix[:,j]), color=f"C{j}", label=f"Arm {int(indxs[j] + 1)}")
                plt.plot(x, y[:,j], color = f"C{j}", linestyle = "--")
    else:
        plt.plot(np.cumsum(rewards), label = f'Rewards', color ="Black")
        for i in range(len(matrix[0,:])):
            plt.plot(np.cumsum(matrix[:,i]), label = f'Arm:{i+1}')
        asplit = int(len(data) / len(arm_means))
        previous = []
        maxes = []
        for j in range(len(arm_means)):
            xmin = j * asplit
            xmax = (j + 1) * asplit
            x = np.column_stack((np.linspace(xmin, xmax, len(data) * 10) for _ in range(len(data[0,:]))))
            if j == 0:
                y = arm_means[j] * x
                maxes.append(xmax)
                previous.append(y[-1])
            else:
                y = (arm_means[j] * (x-maxes[j-1])) + previous[j-1]
                previous.append(y[-1])
            for _ in range(len(data[0,:])):
                plt.plot(x, y[:,_], color = f"C{_}", linestyle = "--")
    plt.xlabel("Steps")
    plt.ylabel("Cumulative sum of rewards")
    plt.title("MultiArmBandits cumulative sum of arms model")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.savefig('1-1MAB-ModelCum.pdf', bbox_inches='tight')
    plt.show()

