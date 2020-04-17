# slope
A spaCy coreference resolution package.

## Install Slope
### Via Pip
Slope is available via pip and has proven to be the easiest way to install Slope.
```bash
$ pip install slope
```

### Install Slope from source
If you'd like to avoid using pip to install Slope, you can do so from source. The process is:
```bash
$ git clone git@github.com:RyanElliott10/slope.git
$ cd slope
$ pip install -e .
```

## Training
If you'd like to train your own model, you'll have to understand the file hierarchy. While not explicitly here in the repo, training uses a `data` directory outside of the main package. i.e. If the path to `coref/` is `./slope/models/coref`, then the path to `data` would be `./data/`. The model was trained on the [PreCo dataset](https://preschool-lab.github.io/PreCo/) for coreference resolution.
