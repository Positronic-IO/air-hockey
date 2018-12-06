''' split input frames in this directory into train & test set'''
import glob, os

current_dir = os.path.dirname(os.path.abspath(__file__))
path_data = 'data/puck/'

# Percentage of images to be used for the test set
percentage_test = 20;

file_train = open('train.txt', 'w')  
file_test = open('test.txt', 'w')

counter = 1  
index_test = round(100 / percentage_test)  
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):  
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))

    if counter == index_test:
        counter = 1
        file_test.write(path_data + title + '.jpg' + "\n")
    else:
        file_train.write(path_data + title + '.jpg' + "\n")
        counter = counter + 1
