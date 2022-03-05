from src.datasets import TEDLIUMCustom, MultiLingualSpokenWordsEnglish, CTRLF_DatasetWrapper, DATASET_MLCOMMONS_PATH, DATASET_TEDLIUM_PATH, KeywordsCSVHeaders


PATH_TO_KEYWORDS = "./keywords_archive.csv"

class LabelCreation:

    def __init__(self):
        self.CTRLF_Dataset = CTRLF_DatasetWrapper(path_to_keywords_csv=PATH_TO_KEYWORDS)

    
    ### ----------- Creating Timestamp Labels From Functions ------------ ###

    def create_label_timestamps(self):
        #Go through keywords.csv, compare between audios where the label is found.
        all_sample_ids = self.CTRLF_Dataset.keywords_df[KeywordsCSVHeaders.TED_SAMPLE_ID]
        for sample_id in all_sample_ids:
            Ted_result_dict, MSWC_result_dict= self.CTRLF_Dataset.get_data(sample_id)
            


        



if __name__=="__main__":
    labelCreation = LabelCreation()