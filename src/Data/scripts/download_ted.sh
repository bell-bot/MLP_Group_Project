#!/bin/bash
# Download TEDLIUM Dataset
wget https://openslr.elda.org/resources/51/TEDLIUM_release-3.tgz
echo "Unzipping TEDLIUM Dataset to the parent folder (under Data)"
tar -xf TEDLIUM_release-3.tgz -C ../
