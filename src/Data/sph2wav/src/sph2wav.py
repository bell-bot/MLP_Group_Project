import os
from sphfile import SPHFile


path = '/home/s1837246/mlpractical/MLP_Group_Project/src/Data/TEDLIUM_release-3/data/sph/'  # Path of folder containing .sph files

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
	outpath = os.path.join("output_"+str(i)+".wav")	
	outputfile.append(outpath)
print(folderpath)
print(outputfile)


for i in range(length):
	sph =SPHFile(folderpath[i])
	print(sph.format)
	sph.write_wav(outputfile[i], 0, 123.57 ) # Customize the period of time to crop



	
	
