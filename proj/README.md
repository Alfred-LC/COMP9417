# README
## Environment
This whole project should be executed in linux based system, because the project uses TensorFlow Decision Forests(tfdf) library, which
only has linux distribution, any attempts to run this project on Windows based system would likely to fail when intsalling tfdf library.
Python 3 is required for this project.
# All neccessary libraries can be installed via following command.
```sh
pip3 install -r requirements.txt
```
## Usage
In order to successfully run the project, you should only run main.py to load and perform model training and viewing model summaries. 
If you need to manipulate model summary and executing order, please modify the code at the bottom of main.py.
The command to run main.py is as follow
```sh
python3 main.py
```