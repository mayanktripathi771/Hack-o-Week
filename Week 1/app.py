from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
import plotly

app = Flask(__name__)

def generate_data():
    # Use explicit Hour offset to avoid pandas string-frequency parsing issues
    hours = pd.date_range(end=pd.Timestamp.now(), periods=168, freq=pd.offsets.Hour())
    usage = np.random.randint(50, 150, size=168)

    df = pd.DataFrame({
        'Hour': hours,
        'Usage': usage
    })

    # Moving Average
    df['Moving_Avg'] = df['Usage'].rolling(window=5).mean()

    # Regression
    df['Hour_Index'] = range(len(df))
    X = df[['Hour_Index']]
    y = df['Usage']

    model = LinearRegression()
    model.fit(X, y)

    df['Predicted'] = model.predict(X)

    return df

@app.route('/')
def index():
    df = generate_data()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Hour'],
        y=df['Usage'],
        mode='lines',
        name='Actual Usage'
    ))

    fig.add_trace(go.Scatter(
        x=df['Hour'],
        y=df['Moving_Avg'],
        mode='lines',
        name='Moving Average'
    ))

    fig.add_trace(go.Scatter(
        x=df['Hour'],
        y=df['Predicted'],
        mode='lines',
        name='Predicted Usage'
    ))

    fig.update_layout(
        title='Peak Hour Electricity Usage – Dormitory',
        xaxis_title='Time',
        yaxis_title='Electricity Usage (Units)',
        template='plotly_white'
    )

    graphJSON = plotly.io.to_json(fig)
    return render_template('index.html', graphJSON=graphJSON)

if __name__ == '__main__':
    # Bind to localhost:8080 when running directly so it's easy to test
    app.run(host='127.0.0.1', port=8080, debug=True)
