import json
import os
import matplotlib.pyplot as plt

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def plot_losses(json_files, output_path):
    plt.figure(figsize=(10, 6))
    
    for json_file in json_files:
        data = read_json(json_file)
        
        learning_rate = None
        losses = []
        steps = []
        
        for entry in data:
            if 'loss' in entry and 'step' in entry:
                losses.append(entry['loss'])
                steps.append(entry['step'])
                if learning_rate is None and 'learning_rate' in entry:
                    learning_rate = entry['learning_rate']
        
        plt.plot(steps, losses, label=f'LR: {learning_rate}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss vs Iteration')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to the specified path
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    # Define the paths to your JSON files
    json_files = [
        "/home/erickaaaaa/nice/results/ofa/0.001_4_adam.json",
        "/home/erickaaaaa/nice/results/ofa/0.0001_4_adam.json",
        "/home/erickaaaaa/nice/results/ofa/5e-05_4_adam.json"
    ]
    
    # Define the path where you want to save the plot
    output_path = "/home/erickaaaaa/nice/results/graph"
    
    # Plot the losses and save the plot
    plot_losses(json_files, output_path)
