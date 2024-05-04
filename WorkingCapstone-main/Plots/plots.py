import pandas as pd
import matplotlib.pyplot as plt

file_names = ['15e32b5v.csv', '10e32b5v.csv', '5e32b5v.csv', '30e16b5v.csv', '20e16b5v.csv', '15e16b5v.csv', '10e16b5v.csv', '5e16b5v.csv', '30e8b5v.csv', '20e8b5v.csv', '15e8b5v.csv', '10e8b5v.csv', '5e8b5v.csv', '30e32b3v.csv', '20e32b3v.csv', '15e32b3v.csv', '10e32b3v.csv', '5e32b3v.csv', '30e16b3v.csv', '20e16b3v.csv', '15e16b3v.csv', '10e16b3v.csv', '5e16b3v.csv', '30e8b3v.csv', '20e8b3v.csv', '15e8b3v.csv', '10e8b3v.csv', '5e8b3v.csv', '30e32b2v.csv', '20e32b2v.csv', '15e32b2v.csv', '10e32b2v.csv', '5e32b2v.csv', '30e16b2v.csv', '20e16b2v.csv', '15e16b2v.csv', '10e16b2v.csv', '5e16b2v.csv', '30e8b2v.csv', '20e8b2v.csv', '15e8b2v.csv', '10e8b2v.csv', '5e8b2v.csv', '30e32b1v.csv', '20e32b1v.csv', '15e32b1v.csv', '10e32b1v.csv', '5e32b1v.csv', '30e16b1v.csv', '20e16b1v.csv', '15e16b1v.csv', '10e16b1v.csv', '5e16b1v.csv', '30e8b1v.csv', '20e8b1v.csv', '15e8b1v.csv', '10e8b1v.csv', '5e8b1v.csv', '30e32b4v.csv', '20e32b4v.csv', '15e32b4v.csv', '10e32b4v.csv', '5e32b4v.csv', '30e16b4v.csv', '20e16b4v.csv', '15e16b4v.csv', '10e16b4v.csv', '5e16b4v.csv', '30e8b4v.csv', '20e8b4v.csv', '15e8b4v.csv', '10e8b4v.csv', '5e8b4v.csv', '30e32b5v.csv', '20e32b5v.csv']

split = 6

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

# Loop through each file and plot the accuracy curves for each batch size on the corresponding subplot
for file_name in file_names:
    if '15e' in file_name  and str(split)+"v" in file_name: # only include files with 30 epochs and 1v in the filename
        df = pd.read_csv(file_name)
        epoch = df['epoch']
        accuracy = df['accuracy']
        loss = df['loss']
        val_accuracy = df['val_accuracy']

        parts = file_name.split("e")
        epoch_find = parts[0]
        remaining = parts[1].split("b")
        batch_size = remaining[0]
        vset = remaining[1].split("v")[0]+"0%"
        # Plot the training accuracy curve for each batch size on the first subplot
        ax1.plot(epoch, accuracy, label=f'Batch size {batch_size}')

        # Plot the validation accuracy curve for each batch size on the second subplot
        ax2.plot(epoch, loss, label=f'Batch size {batch_size}')

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Accuracy')
        ax1.set_title(f'Training Accuracy Curves for {epoch_find} Epochs and {vset} validation split')
        ax1.legend()

        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title(f'Loss Curves for {epoch_find} Epochs and {vset} validation split')
        ax2.legend()

        plt.tight_layout() # adjust the spacing between subplots to prevent overlap
        plt.savefig(f'{epoch_find}e_{vset}v.jpg', bbox_inches='tight')
