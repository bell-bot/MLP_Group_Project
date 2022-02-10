import os 
import csv
import copy
import git



#TODO: Clean this file
def get_git_root(path):

        git_repo = git.Repo(path, search_parent_directories=True)
        git_root = git_repo.git.rev_parse("--show-toplevel")
        return git_root



ORIGINAL_TED_TALK_TRAINVAL_PATH = "/group/corpora/public/lipreading/LRS3/trainval"
#Get the absolute path of the script
#TODO: Could make the code more flexible by accepting arguments in the terminal for where the recording examples will be located
TEST_RECORDED_LABELS_PATH = os.path.join(get_git_root(os.getcwd()), 'src' ,'recording_examples')
CSV_HEADER = ["original_recording_id", "example_recording_id",
        "example_category",
        "original_transcript", "example_transcript",  
        "path_to_original", 
        "path_to_example"]

CSV_FILE_PATH= "./recordings_metadata.csv"

AUDIO_EXTENSIONS = [".m4a", ".mp4"]



def read_csv_file(csv_file_path):
        pass
        

def write_csv_file():
        with open(CSV_FILE_PATH, "w") as csv_file:
                w = csv.DictWriter(csv_file, fieldnames=CSV_HEADER)
                w.writeheader()
                dir_list = os.listdir(TEST_RECORDED_LABELS_PATH)
                for name in dir_list:
                        #Check if it is a directory
                        if os.path.isdir(name):
                                example_category_name = os.path.basename(name)
                                current_path = os.path.join(TEST_RECORDED_LABELS_PATH, example_category_name)
                                print(example_category_name)
                                print(current_path)
                                for path, _ ,files in os.walk(current_path):
                                        if files is not None:
                                                for file_name in files:
                                                        row_csv = {}
                                                        unique_folder_id = os.path.basename(path)
                                                        original_file_name = copy.deepcopy(file_name)
                                                        if "_" in file_name:  #In case multiple recordings of same file name exist, they will be followed with underscore and an alphabet. Make sure to remove that when looking for the original file name
                                                                        #Example: 50007_a.m4a or 50007_b.m4a
                                                                original_file_name = file_name.split("_")[0]

                                                        fname_without_extension, extension = os.path.splitext(file_name)
                                                        original_fname_without_extension, _ =  os.path.splitext(original_file_name)
                                                        original_file_path = os.path.join(ORIGINAL_TED_TALK_TRAINVAL_PATH, unique_folder_id, original_fname_without_extension+ ".mp4")
                                                        full_relative_path = os.path.join(TEST_RECORDED_LABELS_PATH, example_category_name, unique_folder_id,fname_without_extension + ".m4a")
                                                        # print(full_relative_path)
                                                        if extension in AUDIO_EXTENSIONS:
                                                                row_csv["original_recording_id"] = generate_recording_ids(file_unique_id=os.path.join(unique_folder_id, original_fname_without_extension),
                                                                prefix="original")
                                                                row_csv["example_recording_id"]  = generate_recording_ids(file_unique_id=os.path.join(unique_folder_id, fname_without_extension), 
                                                                prefix="example_{}".format(example_category_name))
                                                                row_csv["path_to_original"] = original_file_path
                                                                row_csv["path_to_example"] = full_relative_path
                                                                original_transcript = read_text_file(
                                                                        text_file = os.path.join(ORIGINAL_TED_TALK_TRAINVAL_PATH, unique_folder_id, original_fname_without_extension + ".txt"),
                                                                        original_text_file_format=True)
                                                                row_csv["original_transcript"] =  original_transcript
                                                                if example_category_name == "exact_recordings":
                                                                        row_csv["example_transcript"] = original_transcript
                                                                else:
                                                                        row_csv["example_transcript"] = read_text_file(
                                                                                text_file=os.path.join(path,fname_without_extension + ".txt"))
                                                                # print(full_relative_path)
                                                                
                                                                row_csv["example_category"] = example_category_name
                                                                w.writerow(row_csv)
                                                        else:
                                                                pass
                                        

def generate_recording_ids(prefix, file_unique_id):
        return  prefix + "_" + file_unique_id


def read_text_file(text_file, original_text_file_format= False, capitalise_all_words=True):
        text = None
        exact_num_of_lines = 1
        with open(text_file, "r") as f:
                lines = f.readlines()
                text = lines[0]

                if not original_text_file_format:
                        assert len(lines) == exact_num_of_lines, ("Please make sure the number of lines in text file \"{}\"".format(text_file) + " is only {}.".format(exact_num_of_lines))
                else:
                        text= text.split("Text: ")[1].strip()
        if capitalise_all_words:
                return text.upper()
        else:
                return text





#TODO: For samples and exact recordings, check whether the transcriptions are contained in the original, manual transcriptions
def unit_test():
        pass
#TODO: check whether there is an original file for each recorded file
def unit_test2():
        pass

if __name__ == "__main__":
        #Change working directory to where the script is
        os.chdir(TEST_RECORDED_LABELS_PATH) 
        #Generate CSV file
        write_csv_file()


