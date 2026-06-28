# LA84-AI-12

## How to setup eda_walmart.ipynb
*(Carry out the following steps **ONLY IF** this is the first time you're setting up this project)*

1. **Navigate to the root folder of the project**
- Open your terminal/command prompt and ensure that you are in the folder **'AOL_AI'**

2. **Create the virtual environment**
```bash
py -3.10 -m venv venv
```
*(Make sure you have already installed the Python 3.10 interpreter)*

*(Download from Python's official website: **https://www.python.org/downloads/**)*

3. **Activate the virtual environment**
```bash
.\venv\Scripts\activate
```

4. **Navigate to the folder with the Python notebook**
```bash
cd artifacts
```

5. **Upgrade pip and install the required dependencies**
```bash
python -m pip install --upgrade pip
pip install -r artifacts/requirements.txt
```

## How to run the Streamlit application (main.py)
*(Carry out steps 1, 2, 3, and 5 **ONLY IF** this is the first time you're setting up this project)*
1. **Install the required packages for the React components**
```bash
cd app/components/forecast_ui/frontend
npm install
```

2. **Build the React files**
```bash
npm run build
```

3. **Navigate back to the project root directory**
```bash
cd ../../../..
```

4. **Activate the virtual environment**
```bash
.\venv\Scripts\activate
```
*(Haven't made the venv folder? Refer to Step 2 of the **eda_walmart.ipynb** setup. Then, return to Step 5 here)*

5. **Upgrade pip and install the required dependencies**
```bash
python -m pip install --upgrade pip
pip install -r app/requirements.txt
```
*(You do **NOT** need to run this command if you have already installed the dependencies previously)*

6. **Run the Streamlit appliation**
```bash
streamlit run app/main.py
```

Dataset: 'Walmart Sales Prediction - (Best ML Algorithms)' (Walmart.csv)
Link: https://www.kaggle.com/code/yasserh/walmart-sales-prediction-best-ml-algorithms/input
