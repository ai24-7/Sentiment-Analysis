# Sentiment-Analysis

## What is this model about?

You can use this model to predict the sentiment of movie comments. The result which will be demonstrated in a dashboard is either 'Positive' or 'Negative'. 

## How to run the model

If you want to run dashboard locally and you are not familliar with docker, follow instrunctions below:

First, it is recommended to create an isolated environment with :
```bash
conda create --name YOUR_ENV_NAME 
```

Second, you need to install packages using :
```bash
conda install pip
pip install -r requirements.txt
```

Then, you need to run model.py :
```bash
python model.py
```
This will produce model pickle file.

Finally, you can run the dashboard using streamlit command :
```bash
streamlit run app.py
```

However, if you are familliar with docker, you can use the dockerfile. 

First, you need to run model.py file:
```
python model.py
```
Then you just need to build an image using dockerfile and run it through:
```
docker build -t dashboard:v.1 .
docker run -d -p 8501 dashboard:v.1
```

I hope you enjoy running this dashboard.
Good Luck!

