import os
from sphfile import SPHFile

# change code to ur tedium dir
path = '/home/s1837246/mlpractical/MLP_Group_Project/src/Data/TEDLIUM_release-3/data/sph/'  # Path of folder containing .sph files
# set the path to which you want to save it ti
out_path = ""
folder = os.fsencode(path)

filenames = []
folderpath = []
outputfile = []

for file in os.listdir(folder):
    filename = os.fsdecode(file)
    if filename.endswith( ('.sph') ): # whatever file types you're using...
        filenames.append(filename)

length = len(filenames) 


for i in range(length):
	fpath = os.path.join(path+filenames[i])
	folderpath.append(fpath)
	outpath = os.path.join(filenames[i][:-4]+".wav")	
	outputfile.append(outpath)
print(folderpath)
print(outputfile)


for i in range(length):
	sph =SPHFile(folderpath[i])
	print(sph.format)
	sph.write_wav(os.path.join(out_path, outputfile[i]), 0, 123.57 ) # Customize the period of time to crop



	
	
