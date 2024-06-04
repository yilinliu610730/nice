import json
import os
import matplotlib.pyplot as plt

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def plot_losses(json_files, output_path, title, labels):
    plt.figure(figsize=(10, 6))
    
    for json_file, label in zip(json_files, labels):
        data = read_json(json_file)
        
        losses = []
        steps = []
        
        for entry in data:
            if 'loss' in entry and 'step' in entry:
                losses.append(entry['loss'])
                steps.append(entry['step'])
        
        plt.plot(steps, losses, label=label)
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(False)
    
    # Save the plot to the specified path
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    # Define the paths to your JSON files for learning rate graph
    json_files_lr = [
        "/home/erickaaaaa/nice/results/ofa/0.001_1_adam.json",
        "/home/erickaaaaa/nice/results/ofa/0.0001_1_adam.json",
        "/home/erickaaaaa/nice/results/ofa/1e-05_1_adam.json"
    ]
    
    # Define the paths to your JSON files for optimizer graph
    json_files_optimizer = [
        "/home/erickaaaaa/nice/results/ofa/0.0001_1_adam.json",
        "/home/erickaaaaa/nice/results/ofa/0.0001_1_sgd.json"
    ]
    
    # Define the paths to your JSON files for batch size graph
    json_files_batch_size = [
        "/home/erickaaaaa/nice/results/ofa/0.0001_1_adam.json",
        "/home/erickaaaaa/nice/results/ofa/0.0001_2_adam.json",
        "/home/erickaaaaa/nice/results/ofa/0.0001_4_adam.json"
    ]
    
    # Define labels for the graphs
    labels_lr = ["LR: 1e-03", "LR: 1e-04", "LR: 1e-05"]
    labels_optimizer = ["adam", "sgd"]
    labels_batch_size = ["Batch Size: 1", "Batch Size: 2", "Batch Size: 4"]
    
    # Define the path where you want to save the plots
    output_path_lr = "/home/erickaaaaa/nice/results/graph_lr.png"
    output_path_optimizer = "/home/erickaaaaa/nice/results/graph_optimizer.png"
    output_path_batch_size = "/home/erickaaaaa/nice/results/graph_batch_size.png"
    
    # Plot the losses and save the plots
    plot_losses(json_files_lr, output_path_lr, 'Loss vs Learning Rate', labels_lr)
    plot_losses(json_files_optimizer, output_path_optimizer, 'Loss vs Optimizer', labels_optimizer)
    plot_losses(json_files_batch_size, output_path_batch_size, 'Loss vs Batch Size', labels_batch_size)
