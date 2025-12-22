import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Read data (skip header row)
data = pd.read_csv('/home/stuwork/MRPC-2025-homework/documents/tracking.csv', 
                   header=None, skiprows=1,  # Skip header
                   names=['t', 'qx', 'qy', 'qz', 'qw'])

# Extract data
t = data['t'].values
qx = data['qx'].values
qy = data['qy'].values
qz = data['qz'].values
qw = data['qw'].values

# Parameters
omega = 0.5
alpha = np.pi / 12

# Calculate B_R_D matrix
def B_R_D(t_val):
    cw = np.cos(omega * t_val)
    sw = np.sin(omega * t_val)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    return np.array([
        [cw, -sw*ca, sw*sa],
        [sw, cw*ca, -cw*sa],
        [0, sa, ca]
    ])

# Calculate end-effector quaternions
q_eff = []

for i, t_i in enumerate(t):
    # Drone body rotation
    rot_drone = R.from_quat([qx[i], qy[i], qz[i], qw[i]])
    W_R_B = rot_drone.as_matrix()
    
    # End-effector rotation in world frame
    W_R_D = W_R_B @ B_R_D(t_i)
    
    # Convert to quaternion
    q = R.from_matrix(W_R_D).as_quat()  # [x, y, z, w]
    
    # Convert to [w, x, y, z] and normalize
    q_final = [q[3], q[0], q[1], q[2]]
    q_final = q_final / np.linalg.norm(q_final)
    
    # Ensure q_w >= 0
    if q_final[0] < 0:
        q_final = [-q for q in q_final]
    
    q_eff.append(q_final)

q_eff = np.array(q_eff)

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot 1: Quaternion components
ax1.plot(t, q_eff[:, 0], 'r-', label='q_w', linewidth=1.5)
ax1.plot(t, q_eff[:, 1], 'g-', label='q_x', linewidth=1.5)
ax1.plot(t, q_eff[:, 2], 'b-', label='q_y', linewidth=1.5)
ax1.plot(t, q_eff[:, 3], 'orange', label='q_z', linewidth=1.5)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Quaternion Value')
ax1.set_title('End-effector Quaternion in World Frame')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-1.1, 1.1)

# Plot 2: Norm verification
norms = np.linalg.norm(q_eff, axis=1)
ax2.plot(t, norms, 'purple', linewidth=1.5, label='Norm')
ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Ideal=1')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Quaternion Norm')
ax2.set_title('Normalization Verification')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.9999, 1.0001)

plt.tight_layout()
plt.savefig('/home/stuwork/MRPC-2025-homework/solutions/problem1/quaternion_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Output summary
print(f"Samples: {len(t)}")
print(f"t range: [{t.min():.3f}, {t.max():.3f}]")
print(f"All q_w >= 0: {np.all(q_eff[:, 0] >= 0)}")
print(f"Norm range: [{norms.min():.10f}, {norms.max():.10f}]")
