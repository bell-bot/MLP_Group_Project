import pandas as pd
from src.datasets import LABELS_LINK_CSV_PATH, KEYWORDS_LINK_CSV_PATH, LabelsCSVHeaders, KeywordsCSVHeaders

import csv


""""
Python script that fixes any issues with the labels/keywords csv file, mainly removing any duplicates (if any) and sorting the labels.
"""
class QuickPostprocessCSV:

    def __init__(self,root=LABELS_LINK_CSV_PATH, sort_by=LabelsCSVHeaders.TED_SAMPLE_ID, fix_quotation_marks=False):
        self.path = root
        self.fix_quotation_marks = fix_quotation_marks
        if fix_quotation_marks:
            self.fix_csv_quotation_marks_issue()
        self.df = pd.read_csv(self.path)
        self.col_sort_by = LabelsCSVHeaders.TED_SAMPLE_ID


    #In case csv file is wrapped as "keyword, id, etc..." in quotation marks, make sure it is saved as csv type
    #i.e) "['hours', 4, 'train', '911Mothers_2010W', 'hours/common_voice_en_197289.opus', 41.835, 42.075937499999995, 0.8460095736033205]" -> remove quotation marks
    #NOTE: Creates a duplicate _temp.csv file
    def fix_csv_quotation_marks_issue(self):
        extension_len = len(".csv")
        path_basename = self.path[:-extension_len] #without extension len
        path_csv_file_to_save = path_basename + "_temp.csv" #create a duplicate file to store in
        with open(path_csv_file_to_save, "w+") as f_w, \
             open(self.path, "r") as f_r:
            csv_w = csv.writer(f_w)
            count = 0
            for line in f_r:
                fields = [field.replace("\"","").strip().rstrip("\'").lstrip("\'") for field in line.split(",")]
                csv_w.writerow(fields)
                count+=1
        #update path to the fixed file
        self.path = path_csv_file_to_save

    #Sort by sample id, and then sort by time stamp start (for labels)
    def sort_by_sample_id(self):
        self.df.sort_values(by=self.col_sort_by, inplace=True)
        self.df.reset_index(drop=True, inplace=True)


    def clean_duplicates(self):
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def save(self, to_new_file=False):
        path_csv_file_to_save = None
        if to_new_file:
            extension_len = len(".csv")
            path_basename = self.path[:-extension_len] #without extension len
            path_csv_file_to_save = path_basename + "_cleaned.csv"
           
        else:
            path_csv_file_to_save = self.path
        
        
        self.df.to_csv(path_csv_file_to_save, index=False)
        print(f"Saved to {path_csv_file_to_save}")

if __name__ == "__main__":
    cleaner = QuickPostprocessCSV(root=LABELS_LINK_CSV_PATH, sort_by= LabelsCSVHeaders.TED_SAMPLE_ID, fix_quotation_marks=True)

    cleaner.sort_by_sample_id()
    print("Sorted!")
    cleaner.clean_duplicates()
    print("Removed duplicates")
    cleaner.save(to_new_file=False)
    print("Done!")
