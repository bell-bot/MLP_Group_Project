# Recording Examples directory
Contains folder structure for the manual recordings done to compare with the original Ted Talk ones.  Currently uses absolute path for example scripts too. 

**IMPORTANT: Please run the python script on a DICE Machine to make it work locally on each machines:**
- Install the following packages on your machine (feel free to create a local environment): `gitpython`, 
then run `python3 generate_csv_from_recordings.py`

## Description of recordings_metadata File
Contains pathing to the original recordings and the example recordings. If new recordings are added, please run the python script to remake the files. Please note that the pathing generated in the CSV file right now is only valid on a **DICE** machine, as the original dataset is found on the University of Edinburgh Informatics network.

### Headers:
- `original_recording_id`: Generated a unique recording id, truncuated from file path for conciseness
- `example_recording_id`: Generated a unique recording id, truncuated from file path for conciseness
- `example_category`: Defines the folder (Named as categories - i.e) samples, exact_recordings, and imperfect_examples that the recorded example belongs to.
  - `samples`: Represents recordings where only a phrase is stated from the original one
  - `exact_recordings`: Represents recording examples where the full sentence in the original one is re-iterated.
  - `imperfect_examples`: Represents hard recording examples sentences that contains key terms from the original recordings, but not exact:
    - Original recording transcription: They went through his files and they didn't find anything **v.s** 
    - Example Recording Transcription: They didn't find the files
- `original_transcript`: Transcription of the original recording
- `example_transcript`: Transcription of the recorded example
- `path_to_original`: Absolute path on the DICE machine to the file
- `path_to_example`: Absolute path to each example, assuming root is the top of the github repository


The example recordings were made in .m4a format, while the original recordings were made in .mp4 format (including video).
