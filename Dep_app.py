#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify
import pickle


# In[ ]:


app = Flask(__name__)


# In[ ]:


# Load the trained model
with open("sarima_mod_dep.pkl", "rb") as f:
    model = pickle.load(f)


# In[ ]:


@app.route("/")
def home():
    return "Reliance Stock Prediction API is Running"


# In[ ]:


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        steps = int(data["steps"])
        # Generate forecast
        forecast = model.forecast(steps=steps)
        return jsonify({
            "prediction": forecast.tolist()
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        })


# In[ ]:


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


# In[ ]:




