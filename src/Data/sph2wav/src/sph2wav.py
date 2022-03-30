import os
from sphfile import SPHFile


path = '/Users/Wassim/Documents/Year 4/MLP/CW3:4/MLP_Group_Project/Data/TEDLIUM_release-3/data/sph/'  # Path of folder containing .sph files
out_path = '/Users/Wassim/Documents/Year 4/MLP/CW3:4/MLP_Group_Project/Data/TEDLIUM_release-3/data/wav/'

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
	outpath = os.path.join(filenames[i]+".wav")	
	outputfile.append(outpath)

folderpath = sorted(folderpath)
outputfile = sorted(outputfile)
print(folderpath)
print(outputfile)

for i in range(length):
	sph =SPHFile(folderpath[i])
	print(sph.format)
	sph.write_wav(os.path.join(out_path, outputfile[i])) # Customize the period of time to crop



	
	
