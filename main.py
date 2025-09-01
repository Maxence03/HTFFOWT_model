import numpy as np
import os
import pandas as pd
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import scipy.signal as signal

class FloatingWindTurbineSimulatorOptimized():
    def __init__(self, initial_conditions=None):
        self.initial_conditions = initial_conditions or { 
            'position': np.zeros(6),
            'velocity': np.zeros(6)
        }
        self.prev_position = np.zeros(6)
        self.t_prev = 0.0
        self.v_history = []
        self.t_history = []
        self.F_history = []
        self.force_components_history = {k: [] for k in ['hydrostatic','radiation','diffraction','viscous','wind','mooring']}

        self.M, self.C, self.Aij_dict, self.X, self.Aij_inf, self.Bij_dict, self.stiffness_data = self.build_matrices_from_file()

        # wind
        self.wind_force_magnitude = 0#.0122
        self.wind_application_height = 0.2-0.0439

        # viscous params
        self.drag_coefficient = 0.5
        #self.Cd_roll = 0.8
        #self.Cd_pitch = 0.8
        self.cylinder_diameter = 0.08
        self.rho_water = 1000

        # placeholders for precomputed kernels
        self.kernel_radiation = {}   # key -> sampled K(tau) on grid
        self.kernel_diffraction = {} # load_comp -> sampled K(tau)
        self._precomputed = False

        # logging frequency
        self.log_every = 50

    # ------------------------------ file reading (unchanged but robust) ------------------------------
    def build_matrices_from_file(self):
        # (Same logic as original but simplified here for brevity; user should keep their file parsing)
        # --- MASS ---
        M = np.zeros((6,6))
        with open("floaterV4great_mass.txt","r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts)>=3:
                    try:
                        i = int(float(parts[0]))-1
                        j = int(float(parts[1]))-1
                        if i==j:
                            M[i,j]=float(parts[2])
                    except:
                        pass
        # --- HYDROSTATIC ---
        C = np.zeros((6,6))
        with open("floaterV4great.hst","r") as f:
            for line in f:
                parts=line.strip().split()
                if len(parts)>=3:
                    try:
                        i=int(float(parts[0]))-1
                        j=int(float(parts[1]))-1
                        if i==j:
                            C[i,j]=float(parts[2])
                    except:
                        pass
        # --- Aij / Bij ---
        Aij_dict={}
        Bij_dict={}
        with open("floaterV4_02_5.1","r") as f:
            for line in f:
                parts=line.strip().split()
                if len(parts)<5: continue
                try:
                    omega=float(parts[0])
                    i=int(parts[1])-1
                    j=int(parts[2])-1
                    a_ij=float(parts[3])*1000
                    b_ij=float(parts[4])*1000*omega
                    key=(i,j)
                    Aij_dict.setdefault(key, {'omega':[], 'A':[]})['omega'].append(omega)
                    Aij_dict[key]['A'].append(a_ij)
                    Bij_dict.setdefault(key, {'omega':[], 'B':[]})['omega'].append(omega)
                    Bij_dict[key]['B'].append(b_ij)
                except:
                    pass
        for key in Aij_dict:
            idx = np.argsort(Aij_dict[key]['omega'])
            Aij_dict[key]['omega']=np.array(Aij_dict[key]['omega'])[idx]
            Aij_dict[key]['A']=np.array(Aij_dict[key]['A'])[idx]
        for key in Bij_dict:
            idx = np.argsort(Bij_dict[key]['omega'])
            Bij_dict[key]['omega']=np.array(Bij_dict[key]['omega'])[idx]
            Bij_dict[key]['B']=np.array(Bij_dict[key]['B'])[idx]

        #A_inf_vector = [0.175,0.175,0.48,0.013,0.08,0.07]
        A_inf_vector = [0.12439893, 0.124381, 0.3887026, 0.00315353, 0.00315815, 0.00172046] #from capytaine
        # Diffraction (keep only load_component==0 direction as original)
        with open("floaterV4_02_5.3","r") as f:
            filtered = [line for line in f if len(line.strip().split())>1 and float(line.strip().split()[1])==0.0]
            columns = ['pulsation', 'wave_direction', 'load_component', 'amplitude', 'phase', 'real', 'imaginary']
            data_rows = []
            for line in filtered:
                parts = line.strip().split()
                if len(parts)==7:
                    try:
                        row = [float(p) for p in parts]
                        data_rows.append(row)
                    except:
                        pass
            X = pd.DataFrame(data_rows, columns=columns)

        # Stiffness data: assume same format as original file parsing
        stiffness_data = {}
        current_dof=None
        current_disp=None
        current_matrix=[]
        with open("stiffness_matrices_vs_displacement.txt","r") as f:
            for line in f:
                line=line.strip()
                if line.startswith("DOF displacement:"):
                    if current_dof is not None and current_disp is not None and current_matrix:
                        stiffness_data[(current_dof, current_disp)] = np.array(current_matrix)
                        current_matrix=[]
                    parts=line.split(", displacement = ")
                    dof_label = parts[0].split(": ")[1].strip()
                    disp_value = float(parts[1])
                    current_dof = dof_label
                    current_disp = disp_value
                elif line:
                    row=[float(val) for val in line.split()]
                    current_matrix.append(row)
            if current_dof is not None and current_disp is not None and current_matrix:
                stiffness_data[(current_dof, current_disp)] = np.array(current_matrix)

        return M, C, Aij_dict, X, A_inf_vector, Bij_dict, stiffness_data

    # ------------------------------ wave kinematics (vectorized) ------------------------------
    def wave_kinematics_vectorized(self, t, x, z_array, wave_params):
        z_array = np.asarray(z_array, dtype=float)
        H = float(wave_params['H'])
        T = float(wave_params['T'])
        L = float(wave_params['L'])
        g = 9.81
        omega = 2*np.pi/T
        k = 2*np.pi/L
        phase = k*x - omega*t
        eta = (H/2)*np.cos(phase - np.pi/2)
        exp_kz = np.exp(k*z_array)
        u = (np.pi*H/T) * exp_kz * np.cos(phase - np.pi/2)
        u_dot = -(2*np.pi**2*H/(T**2)) * exp_kz * np.sin(phase - np.pi/2)
        return eta, u, u_dot
        
    # ------------------------------ precompute kernels for radiation & diffraction ------------------------------
    def precompute_kernels(self, t_max, dt):
        """Precompute memory kernels sampled on grid tau = [0, dt, 2dt, ..., t_max].
        This must be called BEFORE simulation (we need dt and t_span known).
        """
        tau = np.arange(0, t_max+dt, dt)
        self.Bij_windowed_all = {}  # dictionnaire pour stocker tous les Bij_windowed
        # radiation: for each (i,i) build K_ii(tau)
        for key, data in self.Bij_dict.items():
            i,j = key
            if i!=j: continue
            omega = np.array(data['omega'])
            Bij = np.array(data['B'])
            # ensure sorted
            idx = np.argsort(omega)
            omega = omega[idx]
            Bij = Bij[idx]
            # apply a Hanning-like window once (choose passband near typical operating frequencies)
            window = self.custom_hanning_window_rad(omega, passband_min=0.9*2*np.pi, passband_max=3.5*2*np.pi, transition_width=0.5*2*np.pi)
            Bij_windowed = Bij * window
            self.Bij_windowed_all[(i,i)] = Bij_windowed  # stocke dans dict complet
            domega = np.diff(omega, prepend=omega[0])
            # vectorized computation of K(tau) = (2/pi) * integral Bij(omega) cos(omega * tau) domega
            # compute cos(omega[:,None] * tau[None,:]) and weight by Bij*domega
            cos_mat = np.cos(np.outer(omega, tau))  # shape (n_omega, n_tau)
            integrand = (Bij_windowed * domega)[:,None] * cos_mat
            K_tau = (2.0/np.pi) * np.sum(integrand, axis=0)
            self.kernel_radiation[(i,i)] = K_tau

        # diffraction: for each load component compute K_diff(tau) such that F_diff(t) = \int K(tau) * eta(t - tau) dtau
        for load_comp in self.X['load_component'].unique():
            df_i = self.X[self.X['load_component']==load_comp]
            omega = df_i['pulsation'].values
            X_i = df_i['real'].values + 1j * df_i['imaginary'].values
            # windowing once
            window = self.custom_hanning_window_rad(omega, passband_min=0.9*2*np.pi, passband_max=4*2*np.pi, transition_width=0.5*2*np.pi)
            X_i_windowed = X_i * window
            idx = np.argsort(omega)
            omega_sorted = omega[idx]
            X_sorted = X_i_windowed[idx]
            domega = np.diff(omega_sorted, prepend=omega_sorted[0])
            # compute K(tau) = (1/2pi) * integral X(omega) e^{i omega tau} domega  (here tau >=0)
            exp_mat = np.exp(1j * np.outer(omega_sorted, tau))
            integrand = (X_sorted * domega)[:,None] * exp_mat
            Ktau_complex = (1.0/(2*np.pi)) * np.sum(integrand, axis=0)
            # we only need the real part for convolution with real eta
            self.kernel_diffraction[int(load_comp)] = np.real(Ktau_complex)

        self.tau_kernel = tau
        self.dt_kernel = dt
        self._precomputed = True
    
    @staticmethod
    def plot_all_Bij_windowed(self):
        dof_labels = ['Surge', 'Sway', 'Heave']

        plt.figure(figsize=(12, 8))
        for (i, j), Bij_windowed in self.Bij_windowed_all.items():
            if (i != j) or (i>2):
                continue
            # Pour chaque DOF diagonal, récupérer omega associé dans Bij_dict
            omega = np.array(self.Bij_dict[(i,j)]['omega'])
            idx = np.argsort(omega)
            omega = omega[idx]
            Bij_windowed_sorted = Bij_windowed[idx]

            plt.plot(omega/(2*np.pi), Bij_windowed_sorted, label=f'{dof_labels[i]}')  # fréquence en Hz

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Bij_windowed (windowed added mass)')
        plt.title('Windowed Bij diagonal elements for each DOF')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        dof_labels=['Roll', 'Pitch', 'Yaw']
        plt.figure(figsize=(12, 8))
        for (i, j), Bij_windowed in self.Bij_windowed_all.items():
            if (i != j) or i<3:
                continue
            # Pour chaque DOF diagonal, récupérer omega associé dans Bij_dict
            omega = np.array(self.Bij_dict[(i,j)]['omega'])
            idx = np.argsort(omega)
            omega = omega[idx]
            Bij_windowed_sorted = Bij_windowed[idx]

            plt.plot(omega/(2*np.pi), Bij_windowed_sorted, label=f'{dof_labels[i-3]}')  # fréquence en Hz

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Bij_windowed (windowed added mass)')
        plt.title('Windowed Bij diagonal elements for each DOF')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # ------------------------------ window helper ------------------------------
    @staticmethod
    def custom_hanning_window_rad(freqs, passband_min, passband_max, transition_width=0.5):
        window = np.zeros_like(freqs, dtype=float)
        left_trans_start = passband_min - transition_width
        right_trans_end = passband_max + transition_width
        mask_flat = (freqs >= passband_min) & (freqs <= passband_max)
        window[mask_flat] = 1.0
        mask_left = (freqs >= left_trans_start) & (freqs < passband_min)
        if np.any(mask_left):
            x = (freqs[mask_left] - left_trans_start) / (passband_min - left_trans_start)
            window[mask_left] = 0.5 * (1 - np.cos(np.pi * x))
        mask_right = (freqs > passband_max) & (freqs <= right_trans_end)
        if np.any(mask_right):
            x = (freqs[mask_right] - passband_max) / (right_trans_end - passband_max)
            window[mask_right] = 0.5 * (1 + np.cos(np.pi * x))
        return window

    # ------------------------------ hydrostatic, wind, mooring (optimized) ------------------------------
    def hydrostatic_force(self, position):
        # keep simple linear hydrostatic used originally
        F_hydro = -np.diag(self.C) * position
        return F_hydro

    def wind_force(self):
        F = np.zeros(6)
        F[0] = -self.wind_force_magnitude
        F[4] = -self.wind_force_magnitude * self.wind_application_height
        return F

    def mooring_force(self, position):
        # vectorized interpolation using np.searchsorted for each DOF
        dof_labels=['Surge (X)','Sway (Y)','Heave (Z)','Roll (Rx)','Pitch (Ry)','Yaw (Rz)']
        total_stiffness = np.zeros((6,6))
        for i, dof_label in enumerate(dof_labels):
            # collect available displacements for this DOF
            disp_values = np.array(sorted([k[1] for k in self.stiffness_data if k[0]==dof_label]))
            if disp_values.size==0:
                continue
            pos_value = position[i]
            idx = np.searchsorted(disp_values, pos_value)
            if idx==0:
                K = self.stiffness_data[(dof_label, disp_values[0])]
            elif idx>=len(disp_values):
                K = self.stiffness_data[(dof_label, disp_values[-1])]
                #print(f"displacement out of bounds for mooring stiffness computations: {pos_value}")
            else:
                lower = disp_values[idx-1]
                upper = disp_values[idx]
                K_lower = self.stiffness_data[(dof_label, lower)]
                K_upper = self.stiffness_data[(dof_label, upper)]
                weight = (pos_value - lower)/(upper-lower)
                K = (1-weight)*K_lower + weight*K_upper
            total_stiffness += K
        F_mooring = -np.diag(total_stiffness) * position
        return F_mooring

    # ------------------------------ radiation (FFT convolution) ------------------------------
    def radiation_force_at_time(self, velocities):
        """
        velocities : array-like shape (N_steps, 6) sampled at dt (self.dt)
        returns 6-vector of radiation forces at current time (last sample), using FFT convolution of precomputed kernel K(tau)
        """
        if not self._precomputed:
            raise RuntimeError('Kernels not precomputed: call precompute_kernels(t_max, dt) before simulate')
        v_arr = np.asarray(velocities)
        N = v_arr.shape[0]
        rad_force = np.zeros(6)
        for i in range(6):
            key = (i,i)
            if key not in self.kernel_radiation:
                continue
            K = self.kernel_radiation[key]
            # discrete convolution: conv[n] = sum_{m=0..n} K[n-m] * v[m] * dt
            conv = signal.fftconvolve(K, v_arr[:,i], mode='full')[:N]
            conv *= self.dt
            rad_force[i] = -conv[-1]
        return rad_force

    # ------------------------------ diffraction (FFT convolution with eta history) ------------------------------
    def diffraction_force_at_time(self, eta_history):
        """eta_history: array-like sampled at dt, length N. returns 6-vector"""
        if not self._precomputed:
            raise RuntimeError('Kernels not precomputed: call precompute_kernels(t_max, dt) before simulate')
        eta = np.asarray(eta_history)
        N = eta.size
        F_vec = np.zeros(6)
        for load_comp, K in self.kernel_diffraction.items():
            conv = signal.fftconvolve(K, eta, mode='full')[:N]
            conv *= self.dt
            # force associated to load_comp at current time is conv[-1]
            F_vec[load_comp] = conv[-1]
        return F_vec

    # ------------------------------ viscous force (vectorized) ------------------------------
    def viscous_force(self, t, surge_velocity, surge_position, heave_position, wave_params):
        # compute for first cylinder and for the other two using vectorized wave_kinematics
        dz = 30
        surge_position -= 0.121
        z1 = np.linspace(-0.03+heave_position, self.wave_kinematics_vectorized(t, surge_position, 0, wave_params)[0], dz)
        eta, u_vec, _ = self.wave_kinematics_vectorized(t, surge_position, z1, wave_params)
        
        rel = u_vec - surge_velocity
        dF1 = 0.5*self.rho_water*self.drag_coefficient*self.cylinder_diameter * np.abs(rel)*rel
        F1 = np.trapezoid(dF1, z1)
        
        # other two pillars shifted
        surge_pos2 = surge_position + 0.1883
        z2 = np.linspace(-0.03+heave_position, self.wave_kinematics_vectorized(t, surge_pos2, 0, wave_params)[0], dz)
        _, u2, _ = self.wave_kinematics_vectorized(t, surge_pos2, z2, wave_params)
        rel2 = u2 - surge_velocity
        dF2 = 2*0.5*self.rho_water*self.drag_coefficient*self.cylinder_diameter * np.abs(rel2)*rel2
        F2 = np.trapezoid(dF2, z2)
        return F1 + F2
       
       
    def viscous_moment_roll_pitch(self, t, heave_velocity, roll_velocity, pitch_velocity, heave_position, wave_params, surge_position):
        
        """
        # at first we don't consider wave surface elevation
        dz = 30
        # floater positions in meters relative to center of rotation (COG)
        floaters = [
        (-0.121, 0.0),
        (0.059, 0.104),
        (0.059, -0.104)
        ]
        # submerged range z (meters)
        z_CM = 0.0439
        z_start = 0
        z_end = -0.03
        z = np.linspace(z_start + heave_position, z_end + heave_position, dz)

        M_roll = 0.0
        M_pitch = 0.0

        for (x_i, y_i) in floaters:
            # Compute wave velocity at floater position along z
            eta, u_wave, _ = self.wave_kinematics_vectorized(t, x_i + surge_position, z, wave_params)

            # Velocity induced by roll angular velocity at each segment
            V_body_y = roll_velocity * z            # roll velocity component in y
            V_body_z_roll = -roll_velocity * y_i   # roll velocity component in z (ignored for drag along x-y only)

            # For viscous drag, consider velocity component aligned with wave velocity u_wave (assumed x-dir)
            # Here, we can approximate relative velocity along x and y axes by projecting wave and body velocity

            # For simplicity, assume drag acts on horizontal velocity component induced by rotation in y-direction (roll)
            # The body velocity vector relative to wave velocity along x is mostly y component from rotation

            rel_roll = -V_body_y  # water velocity u_wave_y zero

            dF_roll = 0.5 * self.rho_water * self.Cd_roll * self.cylinder_diameter * np.abs(rel_roll) * rel_roll

            F_roll = np.trapz(dF_roll, z)

            # Moment arm is y_i for roll moment
            M_roll += F_roll * y_i

            # Similarly for pitch:
            V_body_x = -pitch_velocity * z          # pitch velocity component in x
            V_body_z_pitch = pitch_velocity * x_i   # pitch velocity component in z (ignored here)

            rel_pitch = u_wave - V_body_x

            dF_pitch = 0.5 * self.rho_water * self.Cd_pitch * self.cylinder_diameter * np.abs(rel_pitch) * rel_pitch

            F_pitch = np.trapz(dF_pitch, z)

            # Moment arm is x_i for pitch moment
            M_pitch += F_pitch * x_i
        """
        Bq_heave = 8.11
        Bq_roll = 0.0187
        Bq_pitch = 0.0198
        F_heave = -Bq_heave*heave_velocity*np.abs(heave_velocity)
        M_roll = -Bq_roll*roll_velocity*np.abs(roll_velocity)
        M_pitch = -Bq_pitch*pitch_velocity*np.abs(pitch_velocity)
        return F_heave, M_roll, M_pitch


    # ------------------------------ equations of motion ------------------------------
    def equations_of_motion(self, t, state, wave_params):
        x = state[:6]
        v = state[6:]
        # append histories (these arrays are sampled at solver's t_eval; we assume fixed-step t_eval)
        self.t_history.append(t)
        self.v_history.append(v.copy())
        # hydrostatic, wind, mooring
        F_hydro = self.hydrostatic_force(x)
        F_wind = self.wind_force()
        F_mooring = self.mooring_force(x)
        # viscous (surge, roll and pitch)
        F_viscous = np.zeros(6)
        F_viscous[0] = self.viscous_force(t, v[0], x[0], x[2], wave_params)
        F_viscous[2], F_viscous[3], F_viscous[4] = self.viscous_moment_roll_pitch(t, v[2], v[3], v[4], x[2], wave_params, x[0])
        # radiation and diffraction require sampled histories -> use precomputed kernels
        # build velocities array and eta array sampled at dt from histories
        # histories include pre-fill zeros if set in simulate()
        velocities_arr = np.array(self.v_history)
        eta_arr = np.array([ self.wave_kinematics_vectorized(tt, x[0], 0, wave_params)[0] for tt in self.t_history ])
        F_radiation = self.radiation_force_at_time(velocities_arr)
        F_diffraction = self.diffraction_force_at_time(eta_arr)
        F_total = F_hydro + F_radiation + F_viscous + F_diffraction + F_wind + F_mooring

        # store
        for k in self.force_components_history:
            if k=='hydrostatic': self.force_components_history[k].append(F_hydro.copy())
            elif k=='radiation': self.force_components_history[k].append(F_radiation.copy())
            elif k=='diffraction': self.force_components_history[k].append(F_diffraction.copy())
            elif k=='viscous': self.force_components_history[k].append(F_viscous.copy())
            elif k=='wind': self.force_components_history[k].append(F_wind.copy())
            elif k=='mooring': self.force_components_history[k].append(F_mooring.copy())
        self.F_history.append(F_total.copy())

        if len(self.t_history) % self.log_every == 0:
            print(f"t = {t:.2f} s")

        ddx = np.zeros(6)
        A = [10,10,0.390,0.00486,0.00491,10] #added mass at the omega_n
        for i in range(6):
            if self.M[i,i]>0:
                ddx[i] = F_total[i] / (self.M[i,i] + self.Aij_inf[i])
        self.prev_position = x.copy()
        return np.concatenate([v, ddx])

    # ------------------------------ simulation wrapper ------------------------------
    def simulate(self, t_span, dt, wave_params):
        # prefill
        pre_t = np.linspace(-1, 0, 20)
        self.t_history = list(pre_t)
        self.v_history = [np.zeros(6) for _ in pre_t]
        for k in self.force_components_history:
            self.force_components_history[k] = [np.zeros(6) for _ in pre_t]
        self.pre_fill_steps = len(pre_t)

        # precompute kernels up to t_max = t_span[1]
        self.dt = dt
        self.precompute_kernels(t_span[1], dt)
        self.plot_all_Bij_windowed(self)
        
        initial_state = np.concatenate([ self.initial_conditions['position'], self.initial_conditions['velocity'] ])
        t_eval = np.linspace(t_span[0], t_span[1], int((t_span[1]-t_span[0])/dt)+1)

        solution = integrate.solve_ivp(lambda t,y: self.equations_of_motion(t,y,wave_params), t_span, initial_state, method='RK45', t_eval=t_eval, max_step=dt, atol=1e-5)
        return solution.t, solution.y[:6], solution.y[6:]

    # ------------------------------ plotting & exporting (unchanged semantics) ------------------------------
    def plot_motion(self, t, positions, velocities):
        dof_names=['Surge','Sway','Heave','Roll','Pitch','Yaw']
        fig, axs = plt.subplots(2,3, figsize=(15,10))
        axs=axs.flatten()
        for i in range(6):
            axs[i].plot(t, positions[i])
            axs[i].set_title(dof_names[i])
            axs[i].grid(True)
        plt.tight_layout()
        plt.show()
        
        

    def export_and_plot_forces(self, output_dir="simulation_outputs"):
        os.makedirs(output_dir, exist_ok=True)
        N_skip = self.pre_fill_steps

        t = np.array(self.t_history[N_skip:])
        for name, history in self.force_components_history.items():
            data = np.array(history[N_skip:])
            df = pd.DataFrame(data, columns=[f'{dof}_force' for dof in ['Surge','Sway','Heave','Roll','Pitch','Yaw']])
            df.insert(0,'time', t)
            df.to_csv(os.path.join(output_dir, f'{name}_force.csv'), index=False)
        F_total = np.array(self.F_history)
        df_total = pd.DataFrame(F_total, columns=[f'{dof}_force' for dof in ['Surge','Sway','Heave','Roll','Pitch','Yaw']])
        df_total.insert(0,'time', t)
        df_total.to_csv(os.path.join(output_dir, 'total_force.csv'), index=False)
        print(f"Exported forces to {output_dir}")

def diagnose_radiation_heave(sim, dt=0.005, t_max=60.0):
    # 0) S'assurer qu'on utilise bien A_inf dans l'EDM
    # >>>> IMPORTANT FIX <<<<
    # Remplacer dans equations_of_motion:
    #    denom = self.M[i,i] + self.Aij_inf[i]
    # et supprimer le vecteur A "maison".
    print("Reminder: ensure equations_of_motion uses self.Aij_inf[i] (not a hard-coded A list).")

    sim.dt = dt
    sim.precompute_kernels(t_max, dt)

    # 1) Inspecter K_zz(tau)
    K = sim.kernel_radiation.get((2,2), None)
    if K is None:
        raise RuntimeError("No heave kernel found in sim.kernel_radiation[(2,2)].")
    tau = sim.tau_kernel
    plt.figure()
    plt.plot(tau, K)
    plt.xlabel("tau [s]"); plt.ylabel("K_heave(tau)")
    plt.title("Radiation kernel K_zz(tau)")
    plt.grid(True); plt.tight_layout(); plt.show()

    # 2) Inversion rapide K -> B(ω) par cos-transform (discret)
    #    B(ω) ≈ (π/2) ∫ K(τ) cos(ωτ) dτ
    w_grid = np.linspace(0.1, 30, 500)   # rad/s
    B_rec = []
    for w in w_grid:
        B_rec.append( (np.pi/2.0) * np.trapezoid(K * np.cos(w*tau), tau) )
    B_rec = np.array(B_rec)

    print(f"[K->B] min(B_rec) = {B_rec.min():.4e}, fraction negative = {(B_rec<0).mean():.2%}")
    plt.figure()
    plt.plot(w_grid/(2*np.pi), B_rec)
    plt.xlabel("Frequency [Hz]"); plt.ylabel("B_rec(ω) [kg/s]")
    plt.title("Recovered radiation damping from K_zz")
    plt.grid(True); plt.tight_layout(); plt.show()

    # 3) Test sinus: puissance moyenne P = - <F_rad * v> doit être >= 0 (dissipatif)
    def avg_power_for(omega, Ncycles=40):
        T = 2*np.pi/omega
        t = np.arange(0, Ncycles*T, dt)
        v = np.zeros((t.size, 6))
        v[:,2] = np.cos(omega*t)   # heave velocity (amplitude arbitraire = 1)
        F = np.zeros((t.size, 6))
        for n in range(t.size):
            F[n,:] = sim.radiation_force_at_time(v[:n+1,:])
        P = -F[:,2] * v[:,2]                 # power extracted by radiation
        return t, P.mean(), P

    Mzz = sim.M[2,2]; Czz = sim.C[2,2]; Ainf = sim.Aij_inf[2]
    omega_n = np.sqrt(Czz/(Mzz + Ainf))
    print(f"ω_n (heave, with A_inf) = {omega_n:.3f} rad/s  |  f_n = {omega_n/(2*np.pi):.3f} Hz")

    for factor in [0.7, 1.0, 1.3]:
        tP, Pmean, Pseries = avg_power_for(factor*omega_n)
        print(f"[Sinus test] ω = {factor:.1f} ω_n  ->  <P> = {Pmean:.4e} W  (should be >= 0)")
        plt.figure()
        plt.plot(tP, Pseries)
        plt.xlabel("t [s]"); plt.ylabel("Instantaneous power -F_rad*v")
        plt.title(f"Power time history at ω={factor:.1f} ω_n")
        plt.grid(True); plt.tight_layout(); plt.show()

    # 4) Free decay énergie (heave seul) : l’énergie mécanique doit décroître monotoniquement
    sim.t_history, sim.v_history = [], []  # reset any residual history
    sim.force_components_history = {k: [] for k in sim.force_components_history}
    sim.F_history = []
    sim.initial_conditions = {'position':[0,0, 0.02, 0,0,0], 'velocity':np.zeros(6)}  # 2 cm heave
    t_span = (0, 40)
    t_eval, pos, vel = sim.simulate(t_span, dt, {'H':0, 'T':1, 'L':1})  # H=0 -> no incident wave

    z = pos[2]
    zdot = vel[2]
    E = 0.5*(Mzz + Ainf)*zdot**2 + 0.5*Czz*z**2
    dE = np.diff(E)
    num_increases = (dE > 1e-12).sum()
    print(f"[Energy check] non-decreasing steps in E: {num_increases} (should be 0 or ~0 within tolerance)")
    plt.figure()
    plt.plot(t_eval, E)
    plt.xlabel("t [s]"); plt.ylabel("Mechanical energy [J]")
    plt.title("Heave free-decay energy with radiation")
    plt.grid(True); plt.tight_layout(); plt.show()
# ------------------------------ quick run example ------------------------------
wave_params1 = {'H': 0.02, 'T': 1/2.31, 'L': 0.3}

output_dir = "Test"
wave_params = wave_params31

if __name__=="__main__":
    sim = FloatingWindTurbineSimulatorOptimized()
    t_span=(0,20)
    dt=0.005
    #diagnose_radiation_heave(sim, dt=0.005, t_max=7.5)
    #wave_params={'H':0.01,'T':1/2.3,'L':0.3}
    t,pos,vel = sim.simulate(t_span, dt, wave_params)
    sim.plot_motion(t, pos, vel)
    sim.export_and_plot_forces(output_dir = output_dir)

# Save positions
dof_names = ['Surge', 'Sway', 'Heave', 'Roll', 'Pitch', 'Yaw']
position_df = pd.DataFrame(pos.T, columns=[f'{name}_pos' for name in dof_names])
position_df['time'] = t
position_df = position_df[['time'] + [f'{name}_pos' for name in dof_names]]

# Compute wave elevation at CG (z=0) using surge displacement
wave_elevation = [
    sim.wave_kinematics_vectorized(tt, pos[0, idx], 0, wave_params)[0]
    for idx, tt in enumerate(t)
]
position_df['wave_elev'] = wave_elevation

# Reorder columns for clarity
position_df = position_df[['time', 'wave_elev'] + [f'{name}_pos' for name in dof_names]]
position_df.to_csv(os.path.join(output_dir, 'displacement.csv'), index=False)

# Save forces
if sim.F_history:
    forces_array = np.array(sim.F_history)
    force_df = pd.DataFrame(forces_array, columns=[f'{name}_force' for name in dof_names])
    force_df['time'] = sim.t_history[20:]
    force_df = force_df[['time'] + [f'{name}_force' for name in dof_names]]
    force_df.to_csv(os.path.join(output_dir, 'forces.csv'), index=False)
    print(f"Results saved to: {output_dir}/displacement.csv and forces.csv")
else:
    print("No force history available to save.")













