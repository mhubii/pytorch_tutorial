import os
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folders', nargs='+', required=True, help='Specify the folders to look for data.')
    args = parser.parse_args()

    # create a data frame to save locations and labels
    df = pd.DataFrame(columns=['file_path', 'label'])

    for idx, folder in enumerate(args.folders):
        files = os.scandir(folder)
        for file in files:
            path = os.path.join(folder, file.name)
            df = df.append({'file_path': path, 'label': idx}, ignore_index=True)

    # save as csv file
    df.to_pickle('files.df')
