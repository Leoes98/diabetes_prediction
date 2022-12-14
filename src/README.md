# Test Execution

### Step 1: Forking the original repository

Log in to Github. Then, go to the original project link and click the "Fork" button. This will copy the entire project to your Github user.

### Step 2: Configure git

Open a Terminal and enter the following commands

```
git config --global user.name "<USER>"
git config --global user.email <EMAIL>
```

### Step 3: Clone the project from your own Github

```
git clone https://github.com/Leoes98/diabetes_prediction.git
```

### Step 4: Install the requirements

```
cd diabetes_prediction/

pip install -r requirements.txt
```

### Step 5: Execute the tests in the environment

```
python ./src/data/make_dataset.py

python ./src/models/train_model.py

python ./src/models/predict_model.py

python ./src/visualization/visualize.py
```

### Step 6: Save changes in the repo

```
git add .

git commit -m "Finished tests"

git push
```

Enter your Github user and Personal Access Token. You can check that the changes have been saved to the repository. You can also see the automated tests in Github Actions thanks to the CML .yaml file stored in the folder .github\workflows.
