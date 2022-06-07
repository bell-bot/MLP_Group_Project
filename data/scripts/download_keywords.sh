#!/bin/bash
# Download the English keyword dataset from MLCommons. Downloads 3 directories
mkdir -p ../Multilingual_Spoken_Words
mkdir -p ../Multilingual_Spoken_Words/alignments
mkdir -p ../Multilingual_Spoken_Words/splits
mkdir -p ../Multilingual_Spoken_Words/audio
echo "Created Directory Multilingual_Spoken_Words"
echo "Downloading Audio files"
wget https://storage.googleapis.com/public-datasets-mswc/audio/en.tar.gz -O audio.tar.gz
echo "Downloading Split files"
wget https://storage.googleapis.com/public-datasets-mswc/splits/en.tar.gz -O splits.tar.gz
echo "Downloading Alignment files"
wget https://storage.googleapis.com/public-datasets-mswc/alignments/en.tar.gz -O alignments.tar.gz
echo "Unzipping alignments folder"
tar -xf alignments.tar.gz -C ../Multilingual_Spoken_Words/alignments
echo "Unzipping splits folder"
tar -xf splits.tar.gz -C ../Multilingual_Spoken_Words/splits
echo "Unzipping audio folder"
tar -xf audio.tar.gz -C ../Multilingual_Spoken_Words/audio
echo "Done! (Remove the zip files downloaded in scripts)"