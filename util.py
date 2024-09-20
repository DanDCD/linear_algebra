import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.renderers.default = "notebook"


# Function to plot a vector
def plot_vector(vector, name, color, start=[0, 0]):
    return go.Scatter(
        x=[start[0], start[0] + vector[0]],  # Start from origin (0, 0)
        y=[start[1], start[1] + vector[1]],
        mode="lines+markers+text",
        marker=dict(size=[0, 10], color=color),
        line=dict(width=3, color=color),
        text=[None, name],
        textposition="top center",
    )


# Function to plot a 3D vector
def plot_vector_3d(vector, name, color):
    return go.Scatter3d(
        x=[0, vector[0]],  # Start from origin (0, 0, 0)
        y=[0, vector[1]],
        z=[0, vector[2]],
        mode="lines+markers+text",
        marker=dict(size=[0, 10], color=color),
        line=dict(width=5, color=color),
        text=[None, name],
        textposition="top center",
    )