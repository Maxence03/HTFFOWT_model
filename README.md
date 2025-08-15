# HTFFOWT_model
1. Overview
This script simulates the 6-DOF motion of a floating wind turbine platform under the combined effects of:

- Hydrostatic restoring forces
- Linear radiation forces (with precomputed impulse-response kernels)
- Linear diffraction forces
- Quadratic viscous drag forces
- Mooring system stiffness forces
- Optional steady wind thrust forces

The code integrates the equations of motion in time using scipy.integrate.solve_ivp, logging motions and forces for later analysis.

2. File Requirements
Before running, the simulator expects hydrodynamic and stiffness data files in the working directory:

Mass matrix: floaterV4_mass.txt
Diagonal entries for each DOF.

Hydrostatic stiffness: floaterV4.hst
Diagonal hydrostatic restoring stiffness for each DOF.

Added mass & damping: floaterV4.1
Frequency-dependent added mass (Aij) and potential damping (Bij) coefficients.

Diffraction: floaterV4.3
Complex diffraction force coefficients versus frequency.

Mooring stiffness curves: stiffness_matrices_vs_displacement.txt
Stiffness matrices tabulated by displacement for each DOF.

3. How to Run
Minimal example (already in the script under if __name__ == "__main__"):

from main_wind_wave_moorpy_opti import FloatingWindTurbineSimulatorOptimized

sim = FloatingWindTurbineSimulatorOptimized()

wave_params = {'H': 0.02, 'T': 1/2.3, 'L': 0.3}  # height (m), period (s), wavelength (m)
t_span = (0, 40)  # simulation time window in seconds
dt = 0.005        # timestep

t, pos, vel = sim.simulate(t_span, dt, wave_params)
sim.plot_motion(t, pos, vel)
sim.export_and_plot_forces(output_dir="simulation_outputs/Test")
Output CSVs are saved with:

displacement.csv: time history of 6-DOF motions

forces.csv: total force history

<component>_force.csv: contribution from each force type

4. Physical Assumptions and Simplifications
Hydrodynamics
Radiation forces: Linear convolution model using precomputed memory kernels from Bij frequency data. Off-diagonal terms ignored; only diagonal (same-DOF) terms are used.
Diffraction forces: Linear convolution of wave elevation history with precomputed diffraction kernels. Only load component 0 is kept (surge direction).
Added mass: Infinite-frequency added mass A_inf is applied only on the diagonal.

Waves
Regular (monochromatic) Airy wave model: wave_kinematics_vectorized computes wave elevation and horizontal velocity from wave height H, period T, and wavelength L. No spectral waves or directional spreading.

Only x-direction propagation is modeled; sway-direction wave kinematics are ignored.

No nonlinear wave-body interaction terms.

Viscous forces
Surge drag: Morison-type quadratic drag integrated along two main columns plus a scaled contribution from the third.

Roll & pitch damping: No hydrodynamic computation; replaced by constant empirical quadratic damping coefficients Bq_roll and Bq_pitch.

Mooring
Mooring restoring force is diagonal and obtained by interpolating precomputed stiffness matrices versus displacement for each DOF.

No dynamic mooring line effects; purely quasi-static stiffness.

Wind
Steady constant thrust magnitude (wind_force_magnitude) and application height are user-specified. No wind spectrum, turbulence, or aerodynamic modeling.

Hydrostatics
Purely linear restoring proportional to displacement, from diagonal entries of C.

5. Numerical Assumptions
Time integration: Explicit RK45 with max_step=dt.
Kernels: Precomputed up to t_max = simulation end time.
Initial conditions: Default initial displacements [0, 0, -0.03, 10°, 10°, 0] and zero velocities.
Pre-fill: Initial histories are zero for 1 second before simulation start to allow convolution terms to initialize.

6. Limitations
Off-diagonal hydrodynamic coupling terms (Aij, Bij) are ignored in radiation.
No nonlinear hydrostatics or large-angle restoring.
No aerodynamic torque modeling; only surge and pitch moment from steady wind thrust.
Regular waves only; irregular seas require modification.
Viscous drag is simplified and may not match experimental values without calibration.

