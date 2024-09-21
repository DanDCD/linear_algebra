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
def plot_vector_3d(vector, name, color, start=[0, 0, 0]):
    return go.Scatter3d(
        x=[start[0], start[0] + vector[0]],  # Start from origin (0, 0, 0)
        y=[start[1], start[1] + vector[1]],
        z=[start[2], start[2] + vector[2]],
        mode="lines+markers+text",
        marker=dict(size=[0, 10], color=color),
        line=dict(width=5, color=color),
        text=[None, name],
        textposition="top center",
    )


# Define the plot_vector function for 2D vectors


def plot_vector(start, vector, name, color):

    return go.Scatter(
        x=[start[0], start[0] + vector[0]],  # From the start position to the vector tip
        y=[start[1], start[1] + vector[1]],
        mode="lines+markers+text",
        marker=dict(size=10, color=color),
        line=dict(width=3, color=color),
        text=[None, name],
        textposition="top center",
    )


# Function to plot basis vectors and a given vector in their coordinate system


def plot_basis_vectors_and_projection(e1, e2, v):

    # Calculate v in the coordinate system of the basis vectors

    projection_x = v[0] * e1

    projection_y = v[1] * e2

    result = projection_x + projection_y

    # we can also use the dot product to calculate the projection (these calculations are equivalent):

    mat = np.array([e1, e2]).T  # Create a matrix with basis vectors as columns

    result = np.dot(mat, v)  # Calculate the projection of v onto the basis vectors

    # Create subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Basis Vectors",
            "Vector representation in terms of basis vectors",
        ),
    )

    # Plot basis vectors in the first subplot

    fig.add_trace(plot_vector([0, 0], e1, "e1", "blue"), row=1, col=1)

    fig.add_trace(plot_vector([0, 0], e2, "e2", "red"), row=1, col=1)

    # Plot the original vector and its projections in the second subplot

    fig.add_trace(plot_vector([0, 0], result, f"{v}", "purple"), row=1, col=2)

    fig.add_trace(
        plot_vector([0, 0], projection_x, f"{v[0]} * e1", "blue"), row=1, col=2
    )

    fig.add_trace(
        plot_vector([0, 0], projection_y, f"{v[1]} * e2", "red"), row=1, col=2
    )

    # Update layout settings

    fig.update_layout(
        title="2D Vectors in terms of basis vectors",
        width=1400,
        height=700,
        showlegend=False,
    )

    # Set axis ranges for both subplots

    fig.update_xaxes(range=[-5, 5], row=1, col=1)

    fig.update_yaxes(range=[-5, 5], row=1, col=1)

    fig.update_xaxes(range=[-5, 5], row=1, col=2)

    fig.update_yaxes(range=[-5, 5], row=1, col=2)

    # Show the plot

    fig.show()


# Function to demonstrate the span of two basis vectors


def plot_span_of_vectors(basis_vector1, basis_vector2, range_val=5, grid_size=20):

    # Generate linear combinations of the basis vectors

    a_values = np.linspace(-range_val, range_val, grid_size)  # Coefficients for v1

    b_values = np.linspace(-range_val, range_val, grid_size)  # Coefficients for v2

    # Generate all linear combinations

    span_points = []

    for a in a_values:

        for b in b_values:

            span_points.append(a * basis_vector1 + b * basis_vector2)

    span_points = np.array(span_points)

    # Create the Plotly plot

    fig = go.Figure()

    # Add the first basis vector

    fig.add_trace(
        go.Scatter(
            x=[0, basis_vector1[0]],
            y=[0, basis_vector1[1]],
            mode="lines+markers",
            name="v1",
            marker=dict(size=10, color="blue"),
            line=dict(width=3, color="blue"),
        )
    )

    # Add the second basis vector

    fig.add_trace(
        go.Scatter(
            x=[0, basis_vector2[0]],
            y=[0, basis_vector2[1]],
            mode="lines+markers",
            name="v2",
            marker=dict(size=10, color="red"),
            line=dict(width=3, color="red"),
        )
    )

    # Add points that represent the span of v1 and v2

    fig.add_trace(
        go.Scatter(
            x=span_points[:, 0],
            y=span_points[:, 1],
            mode="markers",
            name="Span",
            marker=dict(size=5, color="purple", opacity=0.5),
        )
    )

    # Update layout settings

    fig.update_layout(
        title="Span of Two Arbitrary Vectors in 2D",
        xaxis=dict(range=[-range_val, range_val], zeroline=True),
        yaxis=dict(range=[-range_val, range_val], zeroline=True),
        width=700,
        height=700,
        showlegend=True,
        plot_bgcolor="white",
    )

    # Show the plot

    fig.show()


# Define the plot_vector function for 3D vectors
def plot_vector_3d(start, vector, name, color):
    return go.Scatter3d(
        x=[start[0], start[0] + vector[0]],  # From the start position to the vector tip
        y=[start[1], start[1] + vector[1]],
        z=[start[2], start[2] + vector[2]],
        mode="lines+markers+text",
        marker=dict(size=5, color=color),
        line=dict(width=3, color=color),
        text=[None, name],
        textposition="top center",
    )


# Function to plot basis vectors and a given vector in their 3D coordinate system
def plot_basis_vectors_and_projection_3d(e1, e2, e3, v):
    # Calculate v in the coordinate system of the basis vectors
    projection_x = v[0] * e1
    projection_y = v[1] * e2
    projection_z = v[2] * e3
    result = projection_x + projection_y + projection_z

    # we can also use the dot product to calculate the projection (these calculations are equivalent):
    mat = np.array([e1, e2, e3]).T  # Create a matrix with basis vectors as columns
    result = np.dot(mat, v)  # Calculate the projection of v onto the basis vectors

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=(
            "Basis Vectors",
            "Vector representation in terms of basis vectors",
        ),
    )

    # Plot basis vectors in the first subplot
    fig.add_trace(plot_vector_3d([0, 0, 0], e1, "e1", "blue"), row=1, col=1)
    fig.add_trace(plot_vector_3d([0, 0, 0], e2, "e2", "red"), row=1, col=1)
    fig.add_trace(plot_vector_3d([0, 0, 0], e3, "e3", "green"), row=1, col=1)

    # Plot the original vector and its projections in the second subplot
    fig.add_trace(plot_vector_3d([0, 0, 0], result, f"v={v}", "purple"), row=1, col=2)
    fig.add_trace(
        plot_vector_3d([0, 0, 0], projection_x, f"{v[0]} * e1", "blue"), row=1, col=2
    )
    fig.add_trace(
        plot_vector_3d([0, 0, 0], projection_y, f"{v[1]} * e2", "red"), row=1, col=2
    )
    fig.add_trace(
        plot_vector_3d([0, 0, 0], projection_z, f"{v[2]} * e3", "green"), row=1, col=2
    )

    # Update layout settings
    fig.update_layout(
        title="3D Vectors in terms of basis vectors",
        width=1400,
        height=700,
        showlegend=False,
    )

    # Set axis ranges for both subplots
    fig.update_scenes(
        xaxis=dict(range=[-5, 5], zeroline=True),
        yaxis=dict(range=[-5, 5], zeroline=True),
        zaxis=dict(range=[-5, 5], zeroline=True),
        row=1,
        col=1,
    )
    fig.update_scenes(
        xaxis=dict(range=[-5, 5], zeroline=True),
        yaxis=dict(range=[-5, 5], zeroline=True),
        zaxis=dict(range=[-5, 5], zeroline=True),
        row=1,
        col=2,
    )

    # Show the plot
    fig.show()


# Function to demonstrate the span of two vectors in 3D
def plot_span_of_vectors_3d(basis_vector1, basis_vector2, range_val=5, grid_size=20):
    # Generate coefficients for linear combinations of the basis vectors
    a_values = np.linspace(
        -range_val, range_val, grid_size
    )  # Coefficients for basis_vector1
    b_values = np.linspace(
        -range_val, range_val, grid_size
    )  # Coefficients for basis_vector2

    # Generate all points in the span by combining basis vectors
    span_points_x = []
    span_points_y = []
    span_points_z = []

    for a in a_values:
        for b in b_values:
            point = a * basis_vector1 + b * basis_vector2
            span_points_x.append(point[0])
            span_points_y.append(point[1])
            span_points_z.append(point[2])

    # Create the Plotly figure for 3D plot
    fig = go.Figure()

    # Add the first basis vector
    fig.add_trace(
        go.Scatter3d(
            x=[0, basis_vector1[0]],
            y=[0, basis_vector1[1]],
            z=[0, basis_vector1[2]],
            mode="lines+markers",
            name="v1",
            marker=dict(size=5, color="blue"),
            line=dict(width=5, color="blue"),
        )
    )

    # Add the second basis vector
    fig.add_trace(
        go.Scatter3d(
            x=[0, basis_vector2[0]],
            y=[0, basis_vector2[1]],
            z=[0, basis_vector2[2]],
            mode="lines+markers",
            name="v2",
            marker=dict(size=5, color="red"),
            line=dict(width=5, color="red"),
        )
    )

    # Add the span points (a mesh of the plane spanned by v1 and v2)
    fig.add_trace(
        go.Scatter3d(
            x=span_points_x,
            y=span_points_y,
            z=span_points_z,
            mode="markers",
            name="Span",
            marker=dict(size=2, color="purple", opacity=0.5),
        )
    )

    # Update layout settings
    fig.update_layout(
        title="Span of Two Vectors in 3D",
        scene=dict(
            xaxis=dict(range=[-range_val, range_val], zeroline=True),
            yaxis=dict(range=[-range_val, range_val], zeroline=True),
            zaxis=dict(range=[-range_val, range_val], zeroline=True),
        ),
        width=700,
        height=700,
        showlegend=True,
    )

    # Show the plot
    fig.show()


# Function to demonstrate the span of three vectors in 3D
def plot_span_of_three_vectors_3d(
    basis_vector1, basis_vector2, basis_vector3, range_val=5, grid_size=10
):
    # Generate coefficients for linear combinations of the basis vectors
    a_values = np.linspace(-range_val, range_val, grid_size)
    b_values = np.linspace(-range_val, range_val, grid_size)
    c_values = np.linspace(-range_val, range_val, grid_size)

    # Generate all points in the span by combining basis vectors
    span_points_x = []
    span_points_y = []
    span_points_z = []

    for a in a_values:
        for b in b_values:
            for c in c_values:
                point = a * basis_vector1 + b * basis_vector2 + c * basis_vector3
                span_points_x.append(point[0])
                span_points_y.append(point[1])
                span_points_z.append(point[2])

    # Create the Plotly figure for 3D plot
    fig = go.Figure()

    # Add the first basis vector
    fig.add_trace(
        go.Scatter3d(
            x=[0, basis_vector1[0]],
            y=[0, basis_vector1[1]],
            z=[0, basis_vector1[2]],
            mode="lines+markers",
            name="v1",
            marker=dict(size=5, color="blue"),
            line=dict(width=5, color="blue"),
        )
    )

    # Add the second basis vector
    fig.add_trace(
        go.Scatter3d(
            x=[0, basis_vector2[0]],
            y=[0, basis_vector2[1]],
            z=[0, basis_vector2[2]],
            mode="lines+markers",
            name="v2",
            marker=dict(size=5, color="red"),
            line=dict(width=5, color="red"),
        )
    )

    # Add the third basis vector
    fig.add_trace(
        go.Scatter3d(
            x=[0, basis_vector3[0]],
            y=[0, basis_vector3[1]],
            z=[0, basis_vector3[2]],
            mode="lines+markers",
            name="v3",
            marker=dict(size=5, color="green"),
            line=dict(width=5, color="green"),
        )
    )

    # Add the span points (a mesh of the space spanned by v1, v2, and v3)
    fig.add_trace(
        go.Scatter3d(
            x=span_points_x,
            y=span_points_y,
            z=span_points_z,
            mode="markers",
            name="Span",
            marker=dict(size=2, color="purple", opacity=0.5),
        )
    )

    # Update layout settings
    fig.update_layout(
        title="Span of Three Vectors in 3D",
        scene=dict(
            xaxis=dict(range=[-range_val, range_val], zeroline=True),
            yaxis=dict(range=[-range_val, range_val], zeroline=True),
            zaxis=dict(range=[-range_val, range_val], zeroline=True),
        ),
        width=700,
        height=700,
        showlegend=True,
    )

    # Show the plot
    fig.show()
