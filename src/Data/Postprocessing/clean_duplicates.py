import pandas as pd
from src.datasets import LABELS_LINK_CSV_PATH, KEYWORDS_LINK_CSV_PATH, LabelsCSVHeaders, KeywordsCSVHeaders
#TODO! Remove duplicates and sort labels

class QuickPostprocessCSV:

    def __init__(self,root=LABELS_LINK_CSV_PATH, sort_by=LabelsCSVHeaders.TED_SAMPLE_ID):
        self.path = root
        self.df = pd.read_csv(root)
        self.col_sort_by = LabelsCSVHeaders.TED_SAMPLE_ID

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
    cleaner = QuickPostprocessCSV(root=KEYWORDS_LINK_CSV_PATH, sort_by= KeywordsCSVHeaders.TED_SAMPLE_ID)
    cleaner.sort_by_sample_id()
    print("Sorted!")
    cleaner.clean_duplicates()
    print("Removed duplicates")
    cleaner.save(to_new_file=False)
    print("Done!")
