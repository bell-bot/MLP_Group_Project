# Data Structure:
```
+-- Data/
|   +-- Tedlium_release-3/
|       +-- ...
|   +-- Multilingual_Spoken_Words/
|   +-- scripts/
|       +-- download_ted.sh
|       +-- download_keywords.sh
|   +-- ...
```


# Downloading the scripts
Use the scripts to download the datasets (on any machine, Google Cloud, DICE, etc...). They also create a new directory as represented in the Data directory structure, and unzip the downloaded files.


**Please run them in the same directory the files exist in (inside of scripts).**

### To run the script:
make the files executable `chmod 700 script_name.sh`, and then run `./script_name.sh`. You might need to log into the mlp clusters first `ssh mlp1` for DICE machines

## DICE Machines
_These steps are optional. However, in case you need to leave (log out - or you want to leave your DICE machine idle), you can run the longjob commands on DICE to not worry about logging out_. You might not need to use these, so these steps are just extra.

Once you are logged into the mlp clusters (`ssh mlp1`), go to the git repository and find the scripts folder under Data folder. 
1) `ssh mlp1`
2) `cd path/to/scripts/`
3) `chmod 700 script_name.sh`
4) _Optional_: You can use `screen` in order to bring background processes into foreground. This allows you to "save" linux shells so you can bring them back and see the progress
  -  `screen -S screen_name`. For example, `screen -S downloading_ted`.
  -   Please check this link to see how to attach or deattach: https://www.rackaid.com/blog/linux-screen-tutorial-and-how-to/ ,
  -   This is also a useful resouce for helpful commands: https://linuxize.com/post/how-to-use-linux-screen/
5) `longjob -c ./script_name.sh`. You will be prompted to enter your DICE password.
  - In case you didn't use screen, you can use `ps -ef | grep script_name.sh` to see if the process is running (although I recommend using `screen` personally)
6) Done! 
  - _Optional_: You can come back to see progress by using `screen` commands or by checking processes using `ps`.



More info: https://computing.help.inf.ed.ac.uk/how-do-i-leave-job-running
_Notes_: 'nice', which is used to indicate priority of process, wasn't used since the process of downloading might not take long, but for following guidelines, its best to use it for other tasks.


