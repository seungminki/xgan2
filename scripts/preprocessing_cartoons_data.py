import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import seaborn as sns
from tqdm import tqdm


# Create image info dataset
def read_cartoon_dataset(path_cartoons, list_sub_folders):

    df = pd.DataFrame()
    for index in tqdm(list_sub_folders):
        path_folder = os.path.join(path_cartoons,index)
        list_files = [file for file in os.listdir(path_folder) if '.csv' in file ]

        for file in list_files:
            path_csv = os.path.join(path_folder, file)
            df_cartoon = pd.read_csv(path_csv,header=None)
            df_cartoon.columns = ["attribute_name", "index_variant","total_num_variants"]
            df_cartoon["filename"] = index + "/" + file
            df_cartoon = df_cartoon.pivot(index="filename",columns="attribute_name", values="index_variant")

            df = pd.concat([df, df_cartoon])

    return df


def make_df_cartoon_dataset(path_cartoons, list_sub_folders):
    df_cartoon = read_cartoon_dataset(path_cartoons, list_sub_folders)
    df_cartoon = df_cartoon.reset_index()
    df_cartoon['subfolder'] = df_cartoon["filename"].apply(lambda x: x.split('/')[0])
    df_cartoon['filename'] = df_cartoon["filename"].apply(lambda x: x.split('/')[-1])
    df_cartoon.to_csv(path_cartoons + "/cartoon100k.csv.gz", header=True, compression="gzip", index=False)

    return df_cartoon


# Show samples per feature
def show_samples_feature(df_cartoon, col, sample_len):
    unique_col = df_cartoon[col].unique()


    print("unique values of {}: {}".format(col, len(unique_col)))

    for value in unique_col:
        idx = (df_cartoon[col] == value)
        file_names = df_cartoon.loc[idx,["filename","subfolder"]]

        print("total filenames with {} equal to {}: {}".format(col, value, len(file_names)))

        for file in file_names[:sample_len].values:
            print(file)
            print(os.path.join(path_cartoons,str(file[1]), file[0]))
            img=mpimg.imread(os.path.join(path_cartoons,str(file[1]), (file[0].split('.'))[0]+".png"))
            imgplot = plt.imshow(img)
            plt.show()


def show_samples_idx(df_cartoon, idx, sample_len):
    file_names = df_cartoon.loc[idx,["filename","subfolder"]]

    print("unique values of index: {}".format(len(file_names)))

    for file in file_names[:sample_len].values:
        print(file)
        print(os.path.join(path_cartoons,str(file[1]), file[0]))
        img=mpimg.imread(os.path.join(path_cartoons,str(file[1]), (file[0].split('.'))[0]+".png"))
        imgplot = plt.imshow(img)
        plt.show()


if __name__ == "__main__":
    path_cartoons = "/workspace/cartoonset100k"
    list_sub_folders = ["0","1","2","3",
                   "4","5","6","7","8","9"]

    df_cartoon = make_df_cartoon_dataset(path_cartoons, list_sub_folders)
    #df_cartoon = pd.read_csv(path_cartoons + "/cartoon100k.csv.gz")

    #analize cartoons with show_samples_feature and show_samples_idx to get the idx to drop
    facial_hair_delete = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    hair_type_delete = [0, 4, 8, 14, 29, 41, 108, 109, 110]
    glass_type_delete = [0, 1, 8, 9, 10]
    face_color_delete = [0, 1, 2]
    glass_color_delete = [2, 5]
    hair_color_delete = [0, 1, 2, 3, 4]
    eye_color_delete = [1, 2, 3]

    # Drop images which have removed attributes
    idx = (
        (df_cartoon.glasses.isin(glass_type_delete))
        | (df_cartoon.hair.isin(hair_type_delete))
        | (df_cartoon.glasses_color.isin(glass_color_delete))
        | (df_cartoon.hair_color.isin(hair_color_delete))
        | (df_cartoon.facial_hair.isin(facial_hair_delete))
        | (df_cartoon.eye_color.isin(eye_color_delete))
        | (df_cartoon.face_color.isin(face_color_delete))
    )
    df_cartoon_filter_final = df_cartoon.loc[-idx, :].reset_index(drop=True)

    #save cartoons ids to keep
    # df_cartoon_filter_final.to_csv(path_cartoons + "/cartoon100k_limited.csv.gz", header=True, compression="gzip", index=False)

    #save the png files and then execute th shell scripts
    writePath = "filelist.txt"
    df_cartoon_filter_final["filename_png"] = df_cartoon_filter_final["filename"].apply(lambda x: x.split('.')[0])+ '.png'
    df_cartoon_filter_final["filename_png"].to_csv(writePath, header=None, index=None, sep=' ', mode='a')
