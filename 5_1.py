#5_1.py
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from collections import deque
# import random
# import numpy as np

# # --- Configuration ---
# WINDOW_SIZE = 5            # Readings for moving average
# SPIKE_THRESHOLD_FACTOR = 1.5
# HAZARDOUS_THRESHOLD = 90   

# # --- Data ---
# data_history = deque(maxlen=50) 
# alert_data_x, alert_data_y = [], [] 

# # --- Plot Setup ---
# fig, ax = plt.subplots(figsize=(7, 4))
# line, = ax.plot([], [], 'b-', label='PM2.5')
# alert_scatter = ax.scatter([], [], color='red', marker='X', s=100, zorder=5, label='Alert')

# ax.set_title('Minimal Real-time PM2.5 Monitor')
# ax.set_xlabel('Time Step')
# ax.set_ylabel('PM2.5 (µg/m³)')
# ax.set_ylim(0, 220) 
# ax.legend()

# # --- Update Function ---
# def update_plot(frame):
#     # 1. CLEAR PREVIOUS ALERT: Remove the marker from the previous frame.
#     alert_data_x.clear()
#     alert_data_y.clear()

#     # Simulate new data: normal (20-60) or spike (100-200)
#     if random.random() < 0.05:
#         value = random.uniform(100, 200)
#     else:
#         value = random.uniform(20, 60)
        
#     data_history.append(value)

#     # Check for spike
#     if len(data_history) >= WINDOW_SIZE:
#         # Calculate average of the window before the current point
#         current_avg = np.mean(list(data_history)[-WINDOW_SIZE-1:-1])
        
#         # Check for absolute hazard OR relative spike
#         if value > HAZARDOUS_THRESHOLD or value > current_avg * SPIKE_THRESHOLD_FACTOR:
#             print(f"!!! ALERT: Spike detected! Value: {value:.2f} µg/m³")
            
#             # 2. ADD CURRENT ALERT: Record coordinates for the new spike only.
#             alert_data_x.append(len(data_history) - 1)
#             alert_data_y.append(value)

#     # Update plot line and marker
#     x_data = list(range(len(data_history)))
#     line.set_xdata(x_data)
#     line.set_ydata(list(data_history))
    
#     # Update scatter plot with the (cleared or new) alert data
#     alert_scatter.set_offsets(np.c_[alert_data_x, alert_data_y])
    
#     # Adjust X-axis to show the moving window
#     ax.set_xlim(max(0, len(data_history) - 50), len(data_history) + 1)
    
#     return line, alert_scatter,

# # --- Run ---
# # interval=1000ms (1 second update rate)
# ani = FuncAnimation(fig, update_plot, interval=1000, blit=True) 
# print("Starting minimal real-time data stream with temporary alert markers.")
# plt.show()

# Inferences:

# 1. Type of Data Being Analyzed

#  The data represents PM2.5 concentration (Particulate Matter ≤ 2.5 µm), measured in micrograms per cubic meter (µg/m³).

# 2. Nature of the Data

#  The data is a continuous, non-stationary time series, characterized as follows:
# Continuous Stream: Generated indefinitely and analyzed sequentially as it arrives.

# Time Series: Each data point is ordered by time, simulated at 1.5-second intervals.

# Non-Stationary: Its statistical properties (mean and variance) vary over time due to external factors like traffic or weather, making fixed thresholds less reliable than dynamic methods (e.g., moving averages).

# 3. Time Duration for Observation

#  The simulation runs continuously using matplotlib.animation.FuncAnimation until the user manually closes the plot window. Data points are generated at a fixed interval of 1.5 seconds.

# 4. Range of Atmospheric Data (Simulated for Chennai)
#  The program simulates values typical of an urban environment like Chennai:
# Normal Fluctuation: Base values follow a random walk centered around 60 µg/m³, reflecting a typical 'Poor' Air Quality Index.


# Hazardous Threshold: 90 µg/m³ triggers an absolute alert.

# Spike Range: Random, sharp spikes up to 200 µg/m³ simulate extreme pollution events.

# 5. Why the Data Fluctuates

#  Fluctuations in PM2.5 concentrations are primarily driven by:
# Routine Emissions (Traffic/Industry): The random walk mimics gradual variations due to hourly changes in traffic flow and industrial activity.

# Unusual Events (Spikes): Random spikes simulate sudden, unpredictable, hazardous events such as:

# Local fires (e.g., waste burning)

# Atmospheric inversions trapping pollutants near the ground

# Major construction or dust events

# 6. Inferences from the Data
#  The simulation provides actionable insights:
# Hazard Identification: Anomalies are detected using a two-pronged strategy:

# Absolute Threshold: Values ≥ 90 µg/m³ are flagged as an immediate health risk.

# Relative Threshold: Values ≥ 1.5× the recent moving average indicate sudden pollution events.

# Real-Time Visualization: The plot gives an instant view of air quality trends, allowing immediate response to hazardous conditions.

# Health Warnings: Console logs and red 'X' markers provide clear alerts for potential respiratory risks.

#------------------------------------------------------------------------------------------------------
#5_2.py
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from collections import deque
# import random
# import numpy as np
 
# # --- Configuration ---
# WINDOW_SIZE = 10         # Readings for calculating the historical average
# CRITICAL_DO_THRESHOLD = 4.0 # mg/L (below this is hazardous for fish/life)
 
# # --- Data Structures ---
# data_history = deque(maxlen=60) # Store last 60 readings (e.g., last 10 minutes)
# alert_data_x, alert_data_y = [], [] # For plotting the alert marker
 
# # --- Plot Setup ---
# fig, ax = plt.subplots(figsize=(7, 4))
# line, = ax.plot([], [], 'c-', label='Dissolved Oxygen (DO)')
# alert_scatter = ax.scatter([], [], color='red', marker='v', s=120, zorder=5, label='DO Alert')
 
# # Draw the critical threshold line
# ax.axhline(CRITICAL_DO_THRESHOLD, color='r', linestyle='--', linewidth=1, label=f'Critical Threshold ({CRITICAL_DO_THRESHOLD} mg/L)')
 
# ax.set_title('Real-time River Dissolved Oxygen Monitor')
# ax.set_xlabel('Time Step')
# ax.set_ylabel('DO Level (mg/L)')
# ax.set_ylim(0, 10) 
# ax.legend(loc='lower left')
 
# # --- Update Function ---
# def update_plot(frame):
#     # 1. CLEAR PREVIOUS ALERT
#     alert_data_x.clear()
#     alert_data_y.clear()
 
#     # Simulate new data: normal (6-9 mg/L) or sudden drop (0-3 mg/L)
#     if random.random() < 0.03: # 3% chance of a pollution event/sensor failure
#         value = random.uniform(0.5, 3.5)
#         is_spike = True
#     else:
#         value = random.uniform(6.0, 9.0)
#         is_spike = False
        
#     data_history.append(value)
 
#     # 4. Threshold Evaluation & Event Detection
#     is_alert = False
    
#     if len(data_history) >= WINDOW_SIZE:
#         current_do = data_history[-1]
        
#         # Check against the absolute critical threshold
#         if current_do < CRITICAL_DO_THRESHOLD:
#             print(f"!!! CRITICAL ALERT: DO Level dropped to {current_do:.2f} mg/L.")
#             is_alert = True
        
#         # Check for sudden drop relative to the recent past (optional)
#         # recent_avg = np.mean(list(data_history)[-WINDOW_SIZE-1:-1])
#         # if current_do < 0.5 * recent_avg: # If less than half the recent average
#         #     print(f"!!! WARNING: Sudden DO drop detected! Value: {current_do:.2f} mg/L")
#         #     is_alert = True

#     # 5. Alert Visualization
#     if is_alert:
#         alert_data_x.append(len(data_history) - 1)
#         alert_data_y.append(data_history[-1])
 
#     # Update plot line and marker
#     x_data = list(range(len(data_history)))
#     line.set_xdata(x_data)
#     line.set_ydata(list(data_history))
    
#     # Update scatter plot for the alert
#     alert_scatter.set_offsets(np.c_[alert_data_x, alert_data_y])
    
#     # Adjust X-axis to show the moving window (last 60 points)
#     ax.set_xlim(max(0, len(data_history) - 60), len(data_history) + 1)
    
#     # Return updated objects
#     return line, alert_scatter,
 
# # --- Run ---
# # interval=500ms (simulate data coming in quickly)
# ani = FuncAnimation(fig, update_plot, interval=500, blit=True) 
# print("Starting real-time DO stream simulation. Look for red triangles for alerts.")
# plt.show()

##  Inferences for Real-time Water Quality Monitoring

### 1. Type of Data Being Analyzed

# The primary data represents **Dissolved Oxygen (DO)** concentration, measured in milligrams per liter (mg/L). DO is a fundamental indicator of water health; sufficient levels (typically $>5$ mg/L) are essential for supporting aquatic life.

# ### 2. Nature of the Data

# The data constitutes a **Continuous, Non-Stationary Time Series**, analyzed sequentially as it streams from the river sensors.

# * **Continuous Stream:** Generated indefinitely and analyzed every few seconds.
# * **Time Series:** Each data point is ordered by time, simulating a live feed.
# * **Non-Stationary:** DO levels naturally fluctuate based on time of day (photosynthesis), temperature, water flow, and organic load, making dynamic monitoring essential.

# ### 3. Time Duration for Observation

# The system operates on a continuous, indefinite schedule. Data is captured and processed with high frequency (e.g., simulated at 500ms or 1-second intervals), enabling near-instantaneous detection of environmental hazards. The visualization window typically displays the last 60 minutes of data for context.

# ### 4. Range of Water Quality Data (Simulated Context)

# The simulation models values typical for a river environment:

# * **Normal Fluctuation:** Base values usually range between 6.0 and 9.0 mg/L (healthy, oxygen-rich conditions).
# * **Critical Threshold:** **4.0 mg/L** is set as the critical absolute threshold, below which conditions are severely stressful or lethal for many fish species and aquatic organisms.
# * **Spike Range (Drop):** Sudden drops are simulated to fall between 0.5 and 3.5 mg/L, mimicking severe pollution events. 

# ### 5. Why the Data Fluctuates (Drops)

# Fluctuations in DO concentrations are driven by multiple factors, with severe drops indicating pollution:

# * **Routine Variations:** Temperature increases (less soluble oxygen), natural decomposition, or quiet water flow.
# * **Unusual Events (Spikes/Drops):** Sudden, unpredictable, hazardous events such as:
#     * Raw sewage discharge (high BOD consumes oxygen).
#     * Industrial effluent dumps (toxic chemicals or organic overload).
#     * Massive algal bloom decay (microbes consume large amounts of oxygen).

# ### 6. Inferences from the Data

# The simulation provides actionable insights for environmental protection:

# * **Hazard Identification:** Anomalies are detected using an **Absolute Critical Threshold** (DO $\leq$ 4.0 mg/L). This ensures any event immediately threatening aquatic life or ecosystem stability is flagged.
# * **Real-Time Visualization:** The plot gives an instant view of DO trends against the critical line, providing clear alerts (e.g., red markers, geo-spatial icons) for the environmental authority dashboard.
# * **SDG Contribution (SDG 6):** Real-time detection allows authorities to **immediately localize the source** of pollution, enabling rapid cleanup and enforcement action, thereby protecting water quality and public health from contaminated sources.
