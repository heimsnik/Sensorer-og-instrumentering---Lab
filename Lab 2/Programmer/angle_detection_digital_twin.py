import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from matplotlib.animation import FuncAnimation
import matplotlib.widgets as widgets

plt.rcParams['font.size'] = 14  
plt.rcParams['axes.labelsize'] = 16  
plt.rcParams['axes.titlesize'] = 18  
plt.rcParams['xtick.labelsize'] = 14  
plt.rcParams['ytick.labelsize'] = 14  

#Constants
speed_of_sound = 343.0  # Speed of sound in air in m/s
d = 0.156 # Side length of the equilateral triangle in meters
sampling_rate = 31250  # in Hz
duration = 1  # duration of the signal in seconds
theta = np.deg2rad(60) 

#Signal to excite the system
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
signal = np.cos(1000*np.pi*t)*np.e**(-t/1000)

#System setup
k = 1/((2/d)*np.cos(theta/2))
mic_positions = np.array([
    [0, k], #M1
    [-d/2,-k*np.sin(theta)],  #M2
    [d/2,-k*np.sin(theta)]  #M3
])

def simulate_microphone_signals(angle_deg, mic_positions):
    angle_rad = np.deg2rad(angle_deg)
    sound_direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    distances = np.dot(mic_positions, sound_direction) #Here is not checked!
    relative_distances = distances - np.min(distances)
    time_delays = relative_distances / speed_of_sound
    time_delays_samples = np.round(time_delays * sampling_rate).astype(int)

    max_delay = np.max(time_delays_samples)
    
    # Generate signals with relative delays
    signals = []
    for delay in time_delays_samples:
        delayed_signal = np.zeros(len(signal) + max_delay)
        delayed_signal[delay:delay + len(signal)] = signal
        signals.append(delayed_signal)
    return signals

def cross_correlate_signals(signals):
    # Cross-correlation between microphoes, received at different delays
    corr_21 = correlate(signals[1], signals[0], mode='full', method='auto')
    corr_31 = correlate(signals[2], signals[0], mode='full', method='auto')
    corr_32 = correlate(signals[2], signals[1], mode='full', method='auto')
    return corr_21, corr_32, corr_31

def estimate_alpha(l_21, l_32, l_31):
    y = np.sqrt(3)*(l_21+l_31)
    x = (l_21-l_31-2*l_32)
    estimated_alpha = np.arctan2(y,x)
    #Corrections due to the arctan2 functions
    if estimated_alpha<=np.pi:
        return np.degrees(estimated_alpha-np.pi)+360, y/x
    else:
        return np.degrees(estimated_alpha+np.pi)+360, y/x

def plot_angle_resolution(arg_vals,alpha_vals): #gives the resolution of angle in terms of argument in arctan
    plt.title("Oppløsning til estimert innfallsvinkel for alle heltallsvinkler")
    plt.xlabel(r"Simulert argument $\frac{y_k}{x_k}$")
    plt.ylabel(r"Estimert innfallsvinkel, $\hat{\alpha_k}$ $[\degree]}$")
    plt.hlines(alpha_vals,-20,20,color='lightgrey',zorder = 1)
    plt.scatter(arg_vals, alpha_vals,color ='green',zorder = 2,s=8)
    plt.show()

def plot_angle_relationships(true_alphas,estimated_alphas):
    plt.title("Sammenheng mellom sann, estimert og simulert innfallsvinkel")
    plt.xlabel(r"Sann innfallsvinkel, $\alpha_k$ $[\degree]}$")
    plt.ylabel(r"Estimert/simulert innfallsvinkel, $\hat{\alpha}$/$\hat{\alpha_k}$ $[\degree]}$")
    plt.plot([0,360],[0,360],color='lightgrey',zorder=1,label=r"Sanne innfallsvinkler $\alpha$")
    plt.scatter([0,90,140,300],[4.9,92.8,142,306.5],color ='red',zorder = 2,s=8,label=r"Estimerte innfallsvinkler $\hat{\alpha}$")
    plt.scatter(true_alphas,estimated_alphas,color ='orange',zorder = 2,s=8,label=r"Simulerte innfallsvinkler $\hat{\alpha_k}$")
    plt.legend()
    plt.show()

def init_gui(mic_positions, angle_deg):
    fig, ax = plt.subplots()
    # Plot microphone positions
    ax.scatter(mic_positions[:, 0], mic_positions[:, 1], color='red', s=20, label='Microphones')
    # Initial line for the angle
    line, = ax.plot([], [], lw=1, label='Sound direction')
    ax.legend()
    ax.set_aspect('equal', 'box')
    ax.grid(True)

    def update(angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        line.set_data([0, d*np.cos(angle_rad)], [0, d*np.sin(angle_rad)])
        fig.canvas.draw_idle()  # Redraw the current figure
        return line,

    return fig, ax, update

def run_simulation():
    #Simulation
    angles_to_loop = 180
    estimated_alphas = np.array([])
    true_alphas = np.array([])
    arg_vals = np.array([])
    fig, ax, update_func = init_gui(mic_positions, 0)
    plt.show(block=False)

    for alpha in range(2, 2*angles_to_loop,1):
        signals = simulate_microphone_signals(alpha,mic_positions)
        correlations = cross_correlate_signals(signals)
        l_max = [np.argmax(correlation) - (len(signals[0]) - 1) for correlation in correlations]
        estimated_values = estimate_alpha(l_max[0], l_max[1], l_max[2])
        estimated_alphas = np.append(estimated_alphas,estimated_values[0])
        print("Simulation from true incident angle: ",alpha,"\tresulted in xcorr lags: ",l_max," and estimated angle: ",round(estimated_values[0],2))
        true_alphas=np.append(true_alphas,alpha)
        arg_vals=np.append(arg_vals,estimated_values[1])
        update_func(alpha)
        plt.pause(0.05)  # Pause to allow the plot to update visually
    plt.close(fig)

    #plot_angle_resolution(arg_vals,estimated_alphas)
    plot_angle_relationships(true_alphas,estimated_alphas)
    print("Mean deviation from true incindent angle: ",np.mean(abs(true_alphas-estimated_alphas)))

run_simulation()