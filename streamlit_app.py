import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


st.title("X & Y Parallelism and Linearity Analysis")
st.write("""
- **Error in X**: The difference in X position compared to the expected position after shifting.
- **Error in Y**: The difference in Y position compared to the expected position after shifting.
The closer the error is to 0, the straighter the scan.
""")

st.markdown("## Upload Files")
st.write("Please upload both files:")

# Two file uploaders for the Matplotlib and Plotly analyses.
uploaded_file_X = st.file_uploader("Upload Compensation_X_raw file", type=["txt", "csv", "xlsx", "xls"], key="file_X")
uploaded_file_Y = st.file_uploader("Upload Compensation_Y_raw file", type=["txt", "csv", "xlsx", "xls"], key="file_Y")

if uploaded_file_X is not None and uploaded_file_Y is not None:
    # Read X file
    try:
        if uploaded_file_X.name.endswith('.csv'):
            df_X = pd.read_csv(uploaded_file_X)
        elif uploaded_file_X.name.endswith('.txt'):
            df_X = pd.read_csv(uploaded_file_X, sep=';', header=None)
        else:
            df_X = pd.read_excel(uploaded_file_X)
        df_X = df_X.iloc[:, :-1]  # Remove last column if not needed
    except Exception as e:
        st.error(f"Error reading X file: {e}")

    # Read Y file
    try:
        if uploaded_file_Y.name.endswith('.csv'):
            df_Y = pd.read_csv(uploaded_file_Y)
        elif uploaded_file_Y.name.endswith('.txt'):
            df_Y = pd.read_csv(uploaded_file_Y, sep=';', header=None)
        else:
            df_Y = pd.read_excel(uploaded_file_Y)
        df_Y = df_Y.iloc[:, :-1]
    except Exception as e:
        st.error(f"Error reading Y file: {e}")

    # Show data preview
    st.markdown("### Data Preview for X file")
    st.dataframe(df_X.head())
    st.markdown("### Data Preview for Y file")
    st.dataframe(df_Y.head())

    st.markdown("---")
    st.markdown("## Parallelism and Linearity Analysis")
    st.write("Select which analysis you want to view:")
    analysis_choice = st.radio("Choose analysis type:", ("Error in X", "Error in Y"))

    # Hard-coded Values
    base_y = 1949000    # Base Y value
    delta_y = 20000000    # Difference in Y for each row
    base_x = 15042000    # Base X value
    delta_x = 20000000   # Difference in X for each column
    
    if analysis_choice == "Error in X":
        st.write("### Error in X")
        y_positions = [base_y + delta_y * i for i in range(len(df_X))]

        # Row-wise Plot (Linearity)
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        for index, (row, y_pos) in enumerate(zip(df_X.iterrows(), y_positions)):
            ax1.plot(row[1].values, label=f'Y = {y_pos}', marker='o')
        ax1.set_title("Linearity (Error in X)")
        ax1.set_xlabel("Index")
        ax1.set_ylabel("Error in X (in nm)")
        ax1.grid(True)
        ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=5, fontsize="small", frameon=True)
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        st.pyplot(fig1)

        # Column-wise Plot (Parallelism)
        x_positions = [base_x + delta_x * col for col in range(df_X.shape[1])]
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        for col, x_pos in zip(df_X.columns, x_positions):
            ax2.plot(df_X[col].values, label=f'X = {x_pos}', marker='o')
        ax2.set_title("Parallelism (Error in X)")
        ax2.set_xlabel("Row Index")
        ax2.set_ylabel("Error in X (in nm)")
        ax2.grid(True)
        ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=5, fontsize="small", frameon=True)
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        st.pyplot(fig2)

    else:
        st.write("### Error in Y")
        # Hard-coded values for Error in Y analysis:
        y_positions = [base_y + delta_y * i for i in range(len(df_Y))]

        # Row-wise Plot (Parallelism)
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        for index, (row, y_pos) in enumerate(zip(df_Y.iterrows(), y_positions)):
            ax3.plot(row[1].values, label=f'Y = {y_pos}', marker='o')
        ax3.set_title("Parallelism (Error in Y)")
        ax3.set_xlabel("Index")
        ax3.set_ylabel("Error in Y (in nm)")
        ax3.grid(True)
        ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=5, fontsize="small", frameon=True)
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        st.pyplot(fig3)

        # Column-wise Plot (Linearity)
        x_positions = [base_x + delta_x * col for col in range(df_Y.shape[1])]
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        for col, x_pos in zip(df_Y.columns, x_positions):
            ax4.plot(df_Y[col].values, label=f'X = {x_pos:.0f}', marker='o')
        ax4.set_title("Linearity (Error in Y)")
        ax4.set_xlabel("Index")
        ax4.set_ylabel("Error in Y (in nm)")
        ax4.grid(True)
        ax4.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=5, fontsize="small", frameon=True)
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        st.pyplot(fig4)

    st.markdown("---")
    st.markdown("## Ideal vs. Current Case")
    st.write("""
The following interactive Plotly chart compares the Ideal Case (blue-dotted line) versus the Current Case (red-solid line).
""")
    
    # Create the Plotly analysis using the uploaded files (assumed to be Compensation_X_raw and Compensation_Y_raw)
    try:
        # Expected (Ideal) Coordinates (hard-coded)
        Top_LeftX = 15042000; Top_RightX = 815042000
        Top_LeftY = 901949000; Top_RightY = 901949000
        Btm_LeftX = 15042000; Btm_RightX = 815042000
        Btm_LeftY = 1949000; Btm_RightY = 1949000

        # Compute Shifted Coordinates by adding the error from the files.
        Shifted_Top_LeftX = Top_LeftX + df_X.iloc[0, 0]
        Shifted_Top_RightX = Top_RightX + df_X.iloc[0, -1]
        Shifted_Top_LeftY = Top_LeftY + df_Y.iloc[0, 0]
        Shifted_Top_RightY = Top_RightY + df_Y.iloc[0, -1]
        Shifted_Btm_LeftX = Btm_LeftX + df_X.iloc[-1, 0]
        Shifted_Btm_RightX = Btm_RightX + df_X.iloc[-1, -1]
        Shifted_Btm_LeftY = Btm_LeftY + df_Y.iloc[-1, 0]
        Shifted_Btm_RightY = Btm_RightY + df_Y.iloc[-1, -1]

        # Define the shifted polygon vertices (closed)
        shifted_polygon_vertices = [
            (Shifted_Top_LeftX, Shifted_Top_LeftY),
            (Shifted_Top_RightX, Shifted_Top_RightY),
            (Shifted_Btm_RightX, Shifted_Btm_RightY),
            (Shifted_Btm_LeftX, Shifted_Btm_LeftY),
            (Shifted_Top_LeftX, Shifted_Top_LeftY)  # Close the polygon
        ]

        # Define the ideal rectangle vertices (closed)
        ideal_rectangle_vertices = [
            (Top_LeftX, Top_LeftY),
            (Top_RightX, Top_RightY),
            (Btm_RightX, Btm_RightY),
            (Btm_LeftX, Btm_LeftY),
            (Top_LeftX, Top_LeftY)  # Close the rectangle
        ]

        # Create a Plotly figure
        fig_plotly = go.Figure()

        # Add the ideal rectangle (dotted blue line)
        fig_plotly.add_trace(go.Scatter(
            x=[x for x, _ in ideal_rectangle_vertices],
            y=[y for _, y in ideal_rectangle_vertices],
            mode='lines',
            name='Ideal Case',
            line=dict(color='blue', dash='dot', width=2)
        ))

        # Add the shifted polygon (solid red line)
        fig_plotly.add_trace(go.Scatter(
            x=[x for x, _ in shifted_polygon_vertices],
            y=[y for _, y in shifted_polygon_vertices],
            mode='lines',
            name='Current Case',
            line=dict(color='red', width=2)
        ))

        # Add annotations for the shifted distances (dividing by 1000 for scaling)
        annotations = [
            (df_X.iloc[0, 0] / 1000, df_Y.iloc[0, 0] / 1000),      # Top-left
            (df_X.iloc[0, -1] / 1000, df_Y.iloc[0, -1] / 1000),    # Top-right
            (df_X.iloc[-1, -1] / 1000, df_Y.iloc[-1, -1] / 1000),  # Bottom-right
            (df_X.iloc[-1, 0] / 1000, df_Y.iloc[-1, 0] / 1000)     # Bottom-left
        ]
        for (x, y), (dx, dy) in zip(shifted_polygon_vertices[:-1], annotations):
            fig_plotly.add_annotation(
                x=x,
                y=y,
                text=f"Δx={dx:.1f}, Δy={dy:.1f} microns",
                showarrow=True,
                arrowhead=2,
                arrowcolor="white",
                font=dict(size=10, color="white"),
                ax=40,
                ay=-40
            )

        # Customize the layout
        fig_plotly.update_layout(
            title="Ideal Case vs Current Case",
            xaxis_title="X Coordinates",
            yaxis_title="Y Coordinates",
            legend=dict(
                title="Legend",
                x=1.05,
                y=1,
                bordercolor="black",
                borderwidth=1,
                font=dict(size=12)
            ),
            width=1200,
            height=900,
            template="plotly_white",
            margin=dict(l=50, r=200, t=50, b=50)
        )
        fig_plotly.update_yaxes(scaleanchor="x", scaleratio=1)

        if st.button("Show Interactive Analysis"):
            st.plotly_chart(fig_plotly, use_container_width=True)

    except Exception as e:
        st.error(f"Error in Plotly analysis: {e}")

else:
    st.info("Please upload both files to proceed.")
