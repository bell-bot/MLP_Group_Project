import csv
from src.datasets import CTRLF_DatasetWrapper
from tqdm import tqdm
x = CTRLF_DatasetWrapper()
with open("ctrlf_metadata.csv", "w") as f:
    #write header
    csv_w =  csv.writer(f)
    csv_w.writerow(x.COLS_OUTPUT_TED_SIMPLIFIED)
    #sort id in ascending order
    ted_id_set = sorted(list(x.TED_sampleids_in_labels_set))
    for i in tqdm(range(0,len(ted_id_set)),desc="Generating Metadata"):
        sample = ted_id_set[i]
        row = x.get_ted_simplified(sample) #gets Ted talk id and transcripts
        row[0]= str(i) + "_" + x.TED.__getitem__(i)["talk_id"]
        row[1] = row[1].strip()
        csv_w.writerow(row)


