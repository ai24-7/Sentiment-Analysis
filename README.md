# Sentiment-Analysis

## What is this model about?

You can use this model to predict the sentiment of movie comments. The result which will be demonstrated in a dashboard is either 'Positive' or 'Negative'. 

## How to run the model

First, it is recommended to create an isolated environment with :
```
conda create --name YOUR_ENV_NAME 
```

Second, you need to install packages using :
```
conda install pip
pip install -r requirements.txt
```

Then, you need to run model.py :
```
python model.py
```
This will produce model pickle file.

Finally, you can run the dashboard using streamlit command :
```
streamlit run app.py
```

I hope you enjoy running this dashboard.
Good Luck!

