# WIP
Made with python 3.10.18

# Images used for training the model
Taken from the oil detection demo video made for the 4th edition of the CubeDesign competition by INPE

# How to generate masks
After downloading the demo for the competition, create a folder named video in the project root and place the downloaded mp4 inside with the name ManchaSat.mp4, then:
```bash
python3 ./python/gen_images.py && python3 ./python/gen_masks.py
```

# libdevice error
This may not happen all the time, but if it does it's necessary to copy the libdevice.10.bc file where the jupyter file is so it doesn't get lost. Using something like os.environ to set the path before importing tensorflow won't fix the problem since it is a quirk of jupyter wanting to find external files for imports inside it's root folder.

Find libdevice on your env and copy it into the jupyter folder in the project root
```bash
find $CONDA_PREFIX -name libdevice.10.bc
```

# How to compile code
```bash
pip install pyinstaller
```
```bash
pyinstaller --onefile file.py
```
the build folder is not needed to run the compiled binary, just run the actual binary with:
```bash
./file
```
