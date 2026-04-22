import os
import matplotlib.pyplot as plt
def save_figf(dir_sublogs, name_graph):
    path_log = os.path.join("logs", dir_sublogs, name_graph)
    plt.savefig(path_log)
    plt.clf()