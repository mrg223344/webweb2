import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from scipy.stats import multivariate_normal

# ========== Part 1: Configuration & Model Parameters ==========
# These values are from your R models.
st.set_page_config(layout="wide")

# --- GMM Parameters (from 'best_model') ---
GMM_MEANS = np.array([
    [296.0132, 198.2082, 367.9903, 173.9132],
    [686.0488, 250.0383, 520.0700, 175.6890],
    [868.8564, 393.8429, 657.2698, 134.8051],
    [1159.3486, 529.7198, 616.9920, 147.5002],
    [1203.1635, 794.3770, 226.7184, 140.0437]
])

GMM_COVS = np.array([
    [[9707.679, 6916.824, 234.172, 1105.468, 2866.625],
     [6916.824, 16189.395, 1072.304, -2740.564, -2284.724],
     [234.172, 1072.304, 40474.981, 6066.644, -29313.590],
     [1105.468, -2740.564, 6066.644, 22679.650, 13519.166],
     [2866.625, -2284.724, -29313.590, 13519.166, 54560.183]],
    [[5228.561, 1316.665, 1742.813, 1438.857, 1888.079],
     [1316.665, 7818.854, -4952.297, -1869.919, -4375.779],
     [1742.813, -4952.297, 39497.188, 11533.302, 7638.223],
     [1438.857, -1869.919, 11533.302, 48451.657, 4746.372],
     [1888.079, -4375.779, 7638.223, 4746.372, 30788.850]],
    [[13420.793, 6539.838, 2321.702, 1900.078, 1609.258],
     [6539.838, 19901.293, -2851.196, -4699.570, -1034.890],
     [2321.702, -2851.196, 30010.334, 24765.572, 10904.122],
     [1900.078, -4699.570, 24765.572, 43971.075, 12383.910],
     [1609.258, -1034.890, 10904.122, 12383.910, 15437.900]],
    [[19745.8332, 9905.9804, 1642.0103, 909.4647, 1365.0155],
     [9905.9804, 28465.1233, 355.4787, -3126.8194, 1193.3600],
     [1642.0103, 355.4787, 20661.1267, 14455.8679, 2788.2002],
     [909.4647, -3126.8194, 14455.8679, 26085.4450, 8476.5907],
     [1365.0155, 1193.3600, 2788.2002, 8476.5907, 14330.454]]
])

GMM_WEIGHTS = np.array([0.09279636, 0.12323578, 0.06732615, 0.71664171])

# --- GLM Parameters (from 'fit_logit_prob_adj') ---
GLM_COEFS = {
    'intercept': -3.056666834,
    'prob1': 1.987347984,
    'prob2': 2.457281428,
    'prob3': 0.844874532,
    'POD1_5max': 0.001379284
}

# ========== Part 2: Constants and Plotting Data ==========
K = GMM_MEANS.shape[1]  # number of classes
D = GMM_MEANS.shape[0]  # number of days

CLASS_LABELS = [f"Class{i+1}" for i in range(K)]
DAYS_VEC = np.arange(1, D + 1)
YOUDEN_THRESHOLD = 0.0960
PALETTE = {"Class1": "#8B0000", "Class2": "#FF4500", "Class3": "#0066CC", "Class4": "#90EE90", "User Input": "black"}

# Prepare class means for plotting
class_means_df = pd.DataFrame(GMM_MEANS, columns=CLASS_LABELS, index=DAYS_VEC)
class_means_df = class_means_df.reset_index().rename(columns={'index': 'day'})
class_means_df_long = class_means_df.melt(
    id_vars='day',
    var_name='class_modal',
    value_name='pred'
)

# ========== Part 3: Prediction Function ==========
def predict_reoperation_risk(drainages, alpha=0):
    """Calculates reoperation risk based on 5-day drainage volumes."""
    if len(drainages) != D or any(v is None or v < 0 for v in drainages):
        st.error(f"Input must be {D} non-negative numbers for Postoperative Days 1-{D}.")
        return None

    new_data = np.array(drainages)
    pod1_5max = np.max(new_data)

    likelihoods = np.zeros(K)
    for i in range(K):
        likelihoods[i] = multivariate_normal.pdf(new_data, mean=GMM_MEANS[:, i], cov=GMM_COVS[i], allow_singular=True) * GMM_WEIGHTS[i]

    sum_likelihoods = np.sum(likelihoods) + 1e-9
    post_prob_raw = likelihoods / sum_likelihoods
    
    post_prob_smoothed = (post_prob_raw + alpha) / (np.sum(post_prob_raw) + K * alpha)
    
    log_odds = GLM_COEFS['intercept'] + \
               GLM_COEFS['prob1'] * post_prob_smoothed[0] + \
               GLM_COEFS['prob2'] * post_prob_smoothed[1] + \
               GLM_COEFS['prob3'] * post_prob_smoothed[2] + \
               GLM_COEFS['POD1_5max'] * pod1_5max
    
    risk_prob = 1 / (1 + np.exp(-log_odds))

    risk_category = "High Risk" if risk_prob > YOUDEN_THRESHOLD else "Low Risk"
    pred_class_label = CLASS_LABELS[np.argmax(post_prob_smoothed)]

    return {
        'pod1_5max': pod1_5max,
        'posterior_probs': post_prob_smoothed,
        'risk_prob': risk_prob,
        'risk_category': risk_category,
        'pred_class': pred_class_label,
        'user_drains': new_data
    }

# ========== Part 4: Plotting Functions ==========
def create_trajectory_plot(user_drains):
    """Creates the drainage trajectory plot with Altair."""
    user_df = pd.DataFrame({'day': DAYS_VEC, 'pred': user_drains, 'class_modal': 'User Input'})
    combined_df = pd.concat([class_means_df_long, user_df])

    lines = alt.Chart(combined_df).mark_line().encode(
        x=alt.X('day:Q', title="Postoperative Day", axis=alt.Axis(tickCount=D)),
        y=alt.Y('pred:Q', title="Drainage Volume (mL)"),
        color=alt.Color('class_modal:N', title="Trajectory Type", 
                        scale=alt.Scale(domain=list(PALETTE.keys()), range=list(PALETTE.values()))),
        strokeWidth=alt.condition(alt.datum.class_modal == 'User Input', alt.value(3), alt.value(1.5))
    ).properties(title="Drainage Trajectories: Class Means and User Input")

    return lines.interactive()

# ========== Part 5: Streamlit UI ==========
st.title("Risk Calculator for Reoperation of Chyle Leak After Thyroidectomy & Neck Dissection")

with st.container(border=True):
    st.subheader("Model Introduction", anchor=False)
    st.write("""
    In patients undergoing thyroidectomy with concomitant neck dissection, a marked increase in postoperative drainage volume 
    or a transition to pale yellow/milky fluid strongly suggests chyle leak. Once confirmed, conservative management should be initiated without delay. 
    We recommend reassessment of therapeutic efficacy and risk stratification on postoperative day (POD) 5. 
    This calculator utilizes trajectory analysis of the first five days of drainage output to automatically estimate the probability of reoperation. 
    For patients identified as high risk, early escalation of conservative measures (e.g., strict fasting with total parenteral nutrition) is recommended. 
    If drainage remains refractory despite aggressive management, timely surgical intervention should be considered.
    """)
    st.caption("_This tool was developed using Gaussian Mixture Model (GMM)-based trajectory clustering combined with multivariable logistic regression, and its performance was validated with bootstrap correction (corrected AUC: 0.872)._")

with st.sidebar:
    st.header("Input Postoperative Drainage Volumes (mL)")
    drains_input = []
    default_values = [200, 400, 400, 400, 200]
    for i in range(D):
        val = st.number_input(f"POD {i+1}:", min_value=0, value=default_values[i], step=1)
        drains_input.append(val)
    
    calculate_button = st.button("Calculate Risk", type="primary", use_container_width=True)

if calculate_button:
    result = predict_reoperation_risk(drains_input)

    if result:
        st.header("Results", anchor=False)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Computed Metrics", anchor=False)
            metrics_df = pd.DataFrame({
                "Metric": [
                    "POD1-5 Max (mL)",
                    "Probability of Class1",
                    "Probability of Class2",
                    "Probability of Class3",
                    "Probability of Class4",
                    "Reoperation Risk Probability"
                ],
                "Value": [
                    f"{result['pod1_5max']:.2f}",
                    f"{result['posterior_probs'][0]:.3f}",
                    f"{result['posterior_probs'][1]:.3f}",
                    f"{result['posterior_probs'][2]:.3f}",
                    f"{result['posterior_probs'][3]:.3f}",
                    f"{result['risk_prob']:.1%}"
                ]
            })
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)

        with col2:
            st.subheader("Risk Visualization", anchor=False)

            risk_prob = result['risk_prob']
            risk_category = result['risk_category']
            threshold = YOUDEN_THRESHOLD

            # Dynamic color based on risk level
            color = "red" if risk_category == "High Risk" else "green"

            # Display category and probability as metrics
            st.metric(
                label="Calculated Risk Category",
                value=risk_category
            )

            st.metric(
                label="Reoperation Risk Probability",
                value=f"{risk_prob:.1%}",
                help=f"The threshold for 'High Risk' is > {threshold:.1%}."
            )

            # Custom styled progress bar with threshold and value labels
            st.write("Risk Level (0% to 100%)")

            progress_html = f"""
            <div style="
                position: relative;
                width: 100%;
                height: 30px;
                background-color: #f0f0f0;
                border-radius: 5px;
                overflow: hidden;
                border: 1px solid #ddd;
                font-family: Arial, sans-serif;
            ">
                <!-- Filled progress -->
                <div style="
                    position: absolute;
                    width: {risk_prob * 100}%;
                    height: 100%;
                    background-color: {color};
                    transition: width 0.4s ease;
                "></div>
                <!-- Threshold line -->
                <div style="
                    position: absolute;
                    left: {threshold * 100}%;
                    width: 2px;
                    height: 100%;
                    background-color: darkred;
                    z-index: 10;
                "></div>
                <!-- Threshold label -->
                <div style="
                    position: absolute;
                    left: {threshold * 100}%;
                    top: 100%;
                    transform: translateX(-50%);
                    font-size: 12px;
                    color: #333;
                    margin-top: 2px;
                    white-space: nowrap;
                ">
                    Threshold {threshold:.1%}
                </div>
                <!-- Current value label above bar -->
                <div style="
                    position: absolute;
                    left: {risk_prob * 100}%;
                    top: -20px;
                    transform: translateX(-50%);
                    font-size: 12px;
                    color: {color};
                    font-weight: bold;
                    white-space: nowrap;
                ">
                    {risk_prob:.1%}
                </div>
            </div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)

            st.caption(f"Risk is classified as 'High' when the probability exceeds {threshold:.1%}.")

        # Plot drainage trajectory
        st.subheader("Drainage Trajectory Plot", anchor=False)
        trajectory_chart = create_trajectory_plot(result['user_drains'])
        st.altair_chart(trajectory_chart, use_container_width=True)