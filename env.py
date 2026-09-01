import gymnasium as gym
import numpy as np


class DoDFEnv(gym.Env):
    """3D evader-vs-kamikaze simulator with selectable evader dynamics.

    The default blue evader retains the original reduced-order linear model.
    Setting ``more_realistic`` selects an AR3-like nonlinear 6-DOF model while
    preserving the same action and observation interfaces. The red kamikaze
    drone remains the original scripted four-rotor rigid-body model.
    """

    def __init__(self, config=None):
        super().__init__()
        config = config or {}

        self.dt = float(config.get("dt", 0.05))
        self.max_steps = int(config.get("max_steps", 3000))
        self.world_limit = float(config.get("world_limit", np.inf))
        self.terminate_on_world_limit = bool(config.get("terminate_on_world_limit", False))
        self.min_separation = float(config.get("min_separation", 4.0))
        self.desired_follow_distance = float(config.get("desired_follow_distance", 20.0))
        self.start_sep_min = float(config.get("start_sep_min", 60.0))
        self.start_sep_max = float(config.get("start_sep_max", 90.0))
        self.more_realistic = bool(config.get("more_realistic", False))

        # High-level command limits commanded by RL: [u_cmd, theta_cmd, phi_cmd]
        self.u_trim = float(config.get("u_trim", 12.0))
        self.u_cmd_span = float(config.get("u_cmd_span", 8.0))
        # Equation 9.60 attitude limits: pitch is restricted to +/-15 deg and
        # bank is produced by the aileron, with a maximum magnitude of 35 deg.
        self.theta_cmd_max = min(abs(float(config.get("theta_cmd_max", np.deg2rad(15.0)))), np.deg2rad(15.0))
        self.phi_cmd_max = float(config.get("phi_cmd_max", np.deg2rad(35.0)))
        self.phi_cmd_max = min(abs(self.phi_cmd_max), np.deg2rad(35.0))
        self.control_mode = str(config.get("control_mode", "surface"))
        self.action_tau = float(config.get("action_tau", 0.75))
        self.action_change_penalty = float(config.get("action_change_penalty", 1.0))
        self.body_rate_penalty = float(config.get("body_rate_penalty", 0.02))
        # Active controls are [elevator, throttle, aileron]. There is no flap
        # action (Eq. 9.53), and the rudder channel is held at trim (Eq. 9.60).
        self.delta_cmd_span = np.array(config.get("delta_cmd_span", [0.22, 0.25, 0.16]), dtype=np.float32)
        if self.delta_cmd_span.shape != (3,):
            raise ValueError("delta_cmd_span must contain [elevator, throttle, aileron]")

        # Actuator lag/saturation for [delta_e, delta_t, delta_a, delta_r]
        self.act_tau = float(config.get("act_tau", 0.65))
        self.delta_rate_limit = np.array(config.get("delta_rate_limit", [0.45, 0.5, 0.20, 0.0]), dtype=np.float32)
        self.delta_trim = np.array(config.get("delta_trim", [0.0, 0.45, 0.0, 0.0]), dtype=np.float32)
        self.delta_min = np.array(config.get("delta_min", [-0.6, 0.0, -0.6, -0.6]), dtype=np.float32)
        self.delta_max = np.array(config.get("delta_max", [0.6, 1.0, 0.6, 0.6]), dtype=np.float32)

        # Linearized reduced-order dynamics around trim.
        # x_lon = [u, w, q, theta], u_lon = [delta_e, delta_t]
        self.A_lon = np.array(
            config.get(
                "A_lon",
                [
                    [-0.5138, 0.3615, -2.3380, -9.8100],
                    [-0.5721, -2.3009, 24.8903, -0.0189],
                    [0.0519, -0.5520, -0.4988, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
            ),
            dtype=np.float32,
        )
        self.B_lon = np.array(
            config.get(
                "B_lon",
                [
                    [0.0, -38.1332],
                    [0.0, 0.0],
                    [-18.2384, 0.0],
                    [0.0, 0.0],
                ],
            ),
            dtype=np.float32,
        )

        # x_lat = [v, p, r, phi, psi], u_lat = [delta_a, delta_r]
        self.A_lat = np.array(
            config.get(
                "A_lat",
                [
                    [-0.6329, 2.3380, -24.8903, 9.8100, 0.0],
                    [-3.1826, -11.5766, 5.1970, 0.0, 0.0],
                    [3.3703, -0.3352, -6.9172, 0.0, 0.0],
                    [0.0, 1.0, 0.0014, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                ],
            ),
            dtype=np.float32,
        )
        self.B_lat = np.array(
            config.get(
                "B_lat",
                [
                    [0.0, 2.7448],
                    [65.0418, 79.5051],
                    [25.9808, -6.0401],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
            ),
            dtype=np.float32,
        )

        # State feedback gains (simple stabilizing defaults).
        # u_lon_cmd = -K_lon * (x_lon - x_lon_ref), shape K_lon: (2, 4)
        self.K_lon = np.array(
            config.get(
                "K_lon",
                [
                    [0.18, 0.30, 0.35, 0.60],
                    [-0.08, -0.02, 0.00, -0.05],
                ],
            ),
            dtype=np.float32,
        )
        # u_lat_cmd = -K_lat * (x_lat - x_lat_ref), shape K_lat: (2, 5)
        self.K_lat = np.array(
            config.get(
                "K_lat",
                [
                    [0.20, 0.45, 0.20, 0.90, 0.0],
                    [0.12, 0.06, 0.40, 0.05, 0.0],
                ],
            ),
            dtype=np.float32,
        )
        self.x_lon_trim = np.array(config.get("x_lon_trim", [self.u_trim, 0.0, 0.0, 0.0]), dtype=np.float32)
        self.x_lat_trim = np.array(config.get("x_lat_trim", [0.0, 0.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        self.x_lon_min = np.array(config.get("x_lon_min", [3.0, -15.0, -4.0, -1.05]), dtype=np.float32)
        self.x_lon_max = np.array(config.get("x_lon_max", [40.0, 15.0, 4.0, 1.05]), dtype=np.float32)
        self.x_lat_min = np.array(config.get("x_lat_min", [-12.0, -4.5, -4.5, -1.22, -np.pi]), dtype=np.float32)
        self.x_lat_max = np.array(config.get("x_lat_max", [12.0, 4.5, 4.5, 1.22, np.pi]), dtype=np.float32)

        # Optional AR3-like blue-aircraft model. The aerodynamic derivatives
        # supplied by Nuno are used directly; geometry, inertia and the missing
        # control derivatives are explicit configurable engineering estimates.
        self.blue_mass = float(config.get("blue_mass", 25.0))
        self.blue_inertia = np.array(config.get("blue_inertia", [4.0, 4.5, 8.0]), dtype=np.float64)
        if self.blue_inertia.shape != (3,) or np.any(self.blue_inertia <= 0.0):
            raise ValueError("blue_inertia must contain three positive values [Ixx, Iyy, Izz]")
        self.blue_wing_area = float(config.get("blue_wing_area", 1.6407473467))
        self.blue_wingspan = float(config.get("blue_wingspan", 3.50))
        self.blue_mean_chord = float(config.get("blue_mean_chord", 0.47))
        self.blue_air_density = float(config.get("blue_air_density", 1.225))
        self.blue_trim_speed = float(config.get("blue_trim_speed", 23.6111111111))
        self.blue_trim_alpha = float(config.get("blue_trim_alpha", -0.0678926212))
        self.blue_trim_throttle = float(config.get("blue_trim_throttle", 0.7139662384))
        self.blue_max_power = float(config.get("blue_max_power", 2000.0))
        self.blue_prop_efficiency = float(config.get("blue_prop_efficiency", 0.75))
        self.blue_prop_min_speed = float(config.get("blue_prop_min_speed", 8.0))
        self.blue_max_static_thrust = float(config.get("blue_max_static_thrust", 120.0))
        self.blue_dyn_substeps = max(int(config.get("blue_dyn_substeps", 5)), 1)
        self.blue_alpha_stall = abs(float(config.get("blue_alpha_stall", np.deg2rad(15.0))))
        self.blue_stall_drag = float(config.get("blue_stall_drag", 2.0))
        self.blue_CL_min = float(config.get("blue_CL_min", -0.8))
        self.blue_CL_max = float(config.get("blue_CL_max", 1.4))
        self.blue_max_body_speed = float(config.get("blue_max_body_speed", 80.0))
        self.blue_max_body_rate = float(config.get("blue_max_body_rate", np.deg2rad(300.0)))

        self.blue_aero = {
            "CL0": float(config.get("blue_CL0", 0.83718)),
            "CLa": float(config.get("blue_CLa", 5.80233)),
            "CD0": float(config.get("blue_CD0", 0.08832)),
            "CDa": float(config.get("blue_CDa", 0.11115)),
            "Cm0": float(config.get("blue_Cm0", -0.07347)),
            "Cma": float(config.get("blue_Cma", -1.08215)),
            "CLq": float(config.get("blue_CLq", 8.10725)),
            "Cmq": float(config.get("blue_Cmq", -13.30914)),
            "CY0": float(config.get("blue_CY0", 0.0)),
            "CYb": float(config.get("blue_CYb", -0.15523)),
            "Cl0": float(config.get("blue_Cl0", 0.0)),
            "Clb": float(config.get("blue_Clb", -0.00493)),
            "Cn0": float(config.get("blue_Cn0", 0.0)),
            "Cnb": float(config.get("blue_Cnb", 0.04165)),
            "CYp": float(config.get("blue_CYp", 0.06254)),
            "CYr": float(config.get("blue_CYr", 0.09638)),
            "Clp": float(config.get("blue_Clp", -0.65507)),
            "Clr": float(config.get("blue_Clr", 0.19007)),
            "Cnp": float(config.get("blue_Cnp", -0.07998)),
            "Cnr": float(config.get("blue_Cnr", -0.03311)),
            # Estimated derivatives; replace when measured values are available.
            "CLde": float(config.get("blue_CLde", 0.35)),
            "Cmde": float(config.get("blue_Cmde", -1.10)),
            "CYda": float(config.get("blue_CYda", 0.0)),
            "Clda": float(config.get("blue_Clda", 0.12)),
            "Cnda": float(config.get("blue_Cnda", -0.02)),
        }
        if self.more_realistic:
            if min(self.blue_mass, self.blue_wing_area, self.blue_wingspan, self.blue_mean_chord) <= 0.0:
                raise ValueError("AR3-like mass and wing geometry must be positive")
            if not 0.0 < self.blue_prop_efficiency <= 1.0:
                raise ValueError("blue_prop_efficiency must be in (0, 1]")
            # Use a self-consistent AR3-like trim only in the opt-in mode.
            self.u_trim = self.blue_trim_speed
            self.x_lon_trim = np.array(
                [
                    self.blue_trim_speed * np.cos(self.blue_trim_alpha),
                    -self.blue_trim_speed * np.sin(self.blue_trim_alpha),
                    0.0,
                    -self.blue_trim_alpha,
                ],
                dtype=np.float32,
            )
            self.x_lat_trim = np.zeros(5, dtype=np.float32)
            self.delta_trim = np.array([0.0, self.blue_trim_throttle, 0.0, 0.0], dtype=np.float32)

        # RL action in [-1, 1]: [elevator, throttle, aileron]. Flap and rudder
        # are intentionally excluded from the controllable interface.
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Observation (22D):
        # [learner_pos(3), learner_psi(1), x_lon(4), x_lat(5),
        #  ref_pos(3), ref_vel(3), prev_action(3)]

        #learner_pos (3): learner drone inertial position [x, y, z]
        # learner_psi (1): learner yaw angle
        # x_lon (4): longitudinal states [u, w, q, theta]
        # x_lat (5): lateral states [v, p, r, phi, psi]
        # ref_pos (3): reference drone inertial position [x, y, z]
        # ref_vel (3): reference drone inertial velocity [vx, vy, vz]
        # prev_action (3): previous [elevator, throttle, aileron] command
        
        obs_high = np.full(22, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )

        self.learner_pos = np.zeros(3, dtype=np.float32)
        self.learner_psi = 0.0
        self.x_lon = np.zeros(4, dtype=np.float32)
        self.x_lat = np.zeros(5, dtype=np.float32)
        self.delta = np.zeros(4, dtype=np.float32)
        self.filtered_action = np.zeros(3, dtype=np.float32)
        self.blue_last_telemetry = {}

        self.ref_pos = np.zeros(3, dtype=np.float32)
        self.ref_vel = np.zeros(3, dtype=np.float32)
        self.ref_psi = 0.0
        self.ref_turn_rate = float(config.get("ref_turn_rate", np.deg2rad(6.0)))
        self.ref_speed = float(config.get("ref_speed", 36.0))
        self.ref_speed_min = float(config.get("ref_speed_min", 20.0))
        self.ref_speed_max = float(config.get("ref_speed_max", 55.0))
        self.ref_speed_gain = float(config.get("ref_speed_gain", 0.40))
        self.ref_climb_amp = float(config.get("ref_climb_amp", 1.2))
        self.ref_climb_freq = float(config.get("ref_climb_freq", 0.15))
        # self.ref_max_turn_rate = float(config.get("ref_max_turn_rate", np.deg2rad(35.0)))
        self.ref_max_turn_rate = float(config.get("ref_max_turn_rate", np.deg2rad(90.0)))

        self.ref_max_climb_rate = float(config.get("ref_max_climb_rate", 6.0))
        self.ref_lead_time = float(config.get("ref_lead_time", 0.5))
        self.ref_lead_blend = float(config.get("ref_lead_blend", 0.1))
        self.ref_heading_gain = float(config.get("ref_heading_gain", 4.0))
        self.ref_dyn_substeps = int(config.get("ref_dyn_substeps", 5))
        # Physical red quadcopter parameters. The pursuer now has a rigid body,
        # four lagged rotor thrusts, gravity, aerodynamic drag, and rotational
        # inertia instead of independent speed/yaw/climb response filters.
        self.ref_mass = float(config.get("ref_mass", 0.85))
        self.ref_inertia = np.array(config.get("ref_inertia", [0.029, 0.029, 0.053]), dtype=np.float32)
        self.ref_arm_length = float(config.get("ref_arm_length", 0.23))
        self.ref_yaw_moment_coeff = float(config.get("ref_yaw_moment_coeff", 0.018))
        self.ref_motor_tau = float(config.get("ref_motor_tau", 0.06))
        self.ref_max_rotor_thrust = float(config.get("ref_max_rotor_thrust", 22.0))
        self.ref_linear_drag = float(config.get("ref_linear_drag", 0.12))
        self.ref_quadratic_drag = float(config.get("ref_quadratic_drag", 0.008))
        self.ref_velocity_gain = float(config.get("ref_velocity_gain", 1.8))
        self.ref_max_accel = float(config.get("ref_max_accel", 22.0))
        self.ref_max_tilt = min(abs(float(config.get("ref_max_tilt", np.deg2rad(60.0)))), np.deg2rad(60.0))
        self.ref_attitude_kp = np.array(config.get("ref_attitude_kp", [11.0, 11.0, 7.0]), dtype=np.float32)
        self.ref_attitude_kd = np.array(config.get("ref_attitude_kd", [4.0, 4.0, 2.7]), dtype=np.float32)
        self.ref_max_torque = np.array(config.get("ref_max_torque", [2.5, 2.5, 1.2]), dtype=np.float32)
        self.gravity = float(config.get("gravity", 9.81))

        self.ref_attitude = np.zeros(3, dtype=np.float32)  # [roll, pitch, yaw]
        self.ref_body_rates = np.zeros(3, dtype=np.float32)  # [p, q, r]
        self.ref_motor_thrusts = np.zeros(4, dtype=np.float32)
        self.ref_yaw_cmd = 0.0

        self.prev_action = np.zeros(3, dtype=np.float32)
        self.t = 0.0
        self._episode_len = 0

    @staticmethod
    def _rotation_body_to_inertial(phi: float, theta: float, psi: float) -> np.ndarray:
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)
        return np.array(
            [
                [cth * cpsi, sphi * sth * cpsi - cphi * spsi, cphi * sth * cpsi + sphi * spsi],
                [cth * spsi, sphi * sth * spsi + cphi * cpsi, cphi * sth * spsi - sphi * cpsi],
                [-sth, sphi * cth, cphi * cth],
            ],
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(
            [
                self.learner_pos,
                np.array([self.learner_psi], dtype=np.float32),
                self.x_lon,
                self.x_lat,
                self.ref_pos,
                self.ref_vel,
                self.prev_action,
            ]
        ).astype(np.float32)

    def _step_realistic_blue(self) -> np.ndarray:
        """Advance the opt-in AR3-like nonlinear rigid-body model."""
        dt_blue = self.dt / float(self.blue_dyn_substeps)
        velocity_body = np.array(
            [self.x_lon[0], self.x_lat[0], self.x_lon[1]], dtype=np.float64
        )
        body_rates = np.array(
            [self.x_lat[1], self.x_lon[2], self.x_lat[2]], dtype=np.float64
        )
        attitude = np.array(
            [self.x_lat[3], self.x_lon[3], self.learner_psi], dtype=np.float64
        )
        de = float(self.delta[0] - self.delta_trim[0])
        throttle = float(np.clip(self.delta[1], 0.0, 1.0))
        da = float(self.delta[2] - self.delta_trim[2])
        aero = self.blue_aero

        for _ in range(self.blue_dyn_substeps):
            u_b, v_b, w_b = velocity_body
            p_b, q_b, r_b = body_rates
            phi, theta, psi = attitude
            airspeed = max(float(np.linalg.norm(velocity_body)), 0.1)
            rate_speed = max(airspeed, self.blue_prop_min_speed)
            alpha = float(np.arctan2(-w_b, max(u_b, 1e-3)))
            beta = float(np.arcsin(np.clip(v_b / airspeed, -1.0, 1.0)))
            # Nuno's derivatives use the conventional aircraft body frame
            # (z down). This environment uses z up, so conventional p/q/r and
            # aerodynamic roll/pitch/yaw moments have the opposite sign.
            p_hat = -p_b * self.blue_wingspan / (2.0 * rate_speed)
            q_hat = -q_b * self.blue_mean_chord / (2.0 * rate_speed)
            r_hat = -r_b * self.blue_wingspan / (2.0 * rate_speed)

            CL_linear = aero["CL0"] + aero["CLa"] * alpha + aero["CLq"] * q_hat + aero["CLde"] * de
            CL = float(np.clip(CL_linear, self.blue_CL_min, self.blue_CL_max))
            stall_excess = max(abs(alpha) - self.blue_alpha_stall, 0.0)
            CD = max(aero["CD0"] + aero["CDa"] * alpha, 0.02) + self.blue_stall_drag * stall_excess ** 2
            Cm = aero["Cm0"] + aero["Cma"] * alpha + aero["Cmq"] * q_hat + aero["Cmde"] * de
            CY = aero["CY0"] + aero["CYb"] * beta + aero["CYp"] * p_hat + aero["CYr"] * r_hat + aero["CYda"] * da
            Cl = aero["Cl0"] + aero["Clb"] * beta + aero["Clp"] * p_hat + aero["Clr"] * r_hat + aero["Clda"] * da
            Cn = aero["Cn0"] + aero["Cnb"] * beta + aero["Cnp"] * p_hat + aero["Cnr"] * r_hat + aero["Cnda"] * da

            dynamic_pressure = 0.5 * self.blue_air_density * airspeed ** 2
            lift = dynamic_pressure * self.blue_wing_area * CL
            drag = dynamic_pressure * self.blue_wing_area * CD
            side_force = dynamic_pressure * self.blue_wing_area * CY
            xz_speed = max(float(np.hypot(u_b, w_b)), 0.1)
            drag_dir = np.array([-u_b / xz_speed, 0.0, -w_b / xz_speed])
            lift_dir = np.array([-w_b / xz_speed, 0.0, u_b / xz_speed])
            prop_speed = max(airspeed, self.blue_prop_min_speed)
            thrust = min(
                throttle * self.blue_max_power * self.blue_prop_efficiency / prop_speed,
                throttle * self.blue_max_static_thrust,
            )
            force_body = drag * drag_dir + lift * lift_dir + np.array([thrust, side_force, 0.0])
            moment_body = -dynamic_pressure * self.blue_wing_area * np.array(
                [self.blue_wingspan * Cl, self.blue_mean_chord * Cm, self.blue_wingspan * Cn]
            )

            rotation = self._rotation_body_to_inertial(phi, theta, psi).astype(np.float64)
            gravity_body = rotation.T @ np.array([0.0, 0.0, -self.gravity])
            velocity_dot = force_body / self.blue_mass + gravity_body - np.cross(body_rates, velocity_body)
            angular_dot = (
                moment_body - np.cross(body_rates, self.blue_inertia * body_rates)
            ) / self.blue_inertia
            velocity_body += dt_blue * velocity_dot
            body_rates += dt_blue * angular_dot
            velocity_body = np.clip(velocity_body, -self.blue_max_body_speed, self.blue_max_body_speed)
            body_rates = np.clip(body_rates, -self.blue_max_body_rate, self.blue_max_body_rate)

            phi, theta, psi = attitude
            p_b, q_b, r_b = body_rates
            cos_theta = max(abs(float(np.cos(theta))), 1e-3)
            euler_rates = np.array(
                [
                    p_b + np.tan(theta) * (q_b * np.sin(phi) + r_b * np.cos(phi)),
                    q_b * np.cos(phi) - r_b * np.sin(phi),
                    (q_b * np.sin(phi) + r_b * np.cos(phi)) / cos_theta,
                ]
            )
            attitude += dt_blue * euler_rates
            attitude[0:2] = np.clip(attitude[0:2], -np.deg2rad(80.0), np.deg2rad(80.0))
            attitude[2] = (attitude[2] + np.pi) % (2.0 * np.pi) - np.pi
            rotation = self._rotation_body_to_inertial(*attitude).astype(np.float64)
            learner_velocity = rotation @ velocity_body
            self.learner_pos += (dt_blue * learner_velocity).astype(np.float32)

        self.x_lon = np.array([velocity_body[0], velocity_body[2], body_rates[1], attitude[1]], dtype=np.float32)
        self.x_lat = np.array(
            [velocity_body[1], body_rates[0], body_rates[2], attitude[0], attitude[2]], dtype=np.float32
        )
        self.learner_psi = float(attitude[2])
        self.blue_last_telemetry = {
            "telemetry_blue_airspeed_mps": airspeed,
            "telemetry_blue_alpha_rad": alpha,
            "telemetry_blue_beta_rad": beta,
            "telemetry_blue_CL": CL,
            "telemetry_blue_CD": CD,
            "telemetry_blue_thrust_n": thrust,
            "telemetry_blue_shaft_power_w": throttle * self.blue_max_power,
        }
        return learner_velocity.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_len = 0
        self.t = 0.0

        self.learner_pos = self.np_random.uniform(-10.0, 10.0, size=3).astype(np.float32)
        self.learner_psi = float(self.np_random.uniform(-np.pi, np.pi))
        self.x_lon = self.x_lon_trim.copy()
        self.x_lon[0] += self.np_random.uniform(-1.0, 1.0)
        self.x_lat = self.x_lat_trim.copy()
        self.delta = self.delta_trim.copy()
        self.filtered_action = np.zeros(3, dtype=np.float32)
        self.blue_last_telemetry = {}

        start_sep_xy = float(self.np_random.uniform(self.start_sep_min, self.start_sep_max))
        start_bearing = float(self.np_random.uniform(-np.pi, np.pi))
        start_dz = float(self.np_random.uniform(-4.0, 4.0))
        self.ref_pos = self.learner_pos + np.array(
            [start_sep_xy * np.cos(start_bearing), start_sep_xy * np.sin(start_bearing), start_dz],
            dtype=np.float32,
        )
        los0 = self.learner_pos - self.ref_pos
        self.ref_psi = float(np.arctan2(los0[1], los0[0]) + self.np_random.uniform(-0.2, 0.2))
        self.ref_attitude = np.array([0.0, 0.0, self.ref_psi], dtype=np.float32)
        self.ref_body_rates = np.zeros(3, dtype=np.float32)
        self.ref_yaw_cmd = self.ref_psi
        self.ref_motor_thrusts = np.full(4, self.ref_mass * self.gravity / 4.0, dtype=np.float32)
        self.ref_vel = np.array(
            [self.ref_speed * np.cos(self.ref_psi), self.ref_speed * np.sin(self.ref_psi), 0.0],
            dtype=np.float32,
        )
        self.prev_action = np.zeros(3, dtype=np.float32)

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self._episode_len += 1
        self.t += self.dt

        a_raw = np.asarray(action, dtype=np.float32)
        a_raw = np.clip(a_raw, -1.0, 1.0)
        action_alpha = self.dt / max(self.action_tau, self.dt)
        self.filtered_action = self.filtered_action + action_alpha * (a_raw - self.filtered_action)
        a = np.clip(self.filtered_action, -1.0, 1.0)
        last_action = self.prev_action.copy()
        self.prev_action = a

        # 1) RL commands. The default exposes UAV-like control surface deflections:
        # [elevator, throttle, aileron]. Flap is absent and rudder remains at
        # trim. The high-level mode is retained
        # for compatibility with earlier experiments.
        if self.control_mode == "surface":
            delta_cmd = self.delta_trim.copy()
            delta_cmd[:3] += self.delta_cmd_span * a
            delta_cmd[3] = self.delta_trim[3]
        else:
            u_cmd = self.u_trim + self.u_cmd_span * float(a[0])
            theta_cmd = self.theta_cmd_max * float(a[1])
            phi_cmd = self.phi_cmd_max * float(a[2])
            x_lon_ref = np.array([u_cmd, 0.0, 0.0, theta_cmd], dtype=np.float32)
            x_lat_ref = np.array([0.0, 0.0, 0.0, phi_cmd, 0.0], dtype=np.float32)
            u_lon_cmd = -self.K_lon @ (self.x_lon - x_lon_ref)     # [delta_e, delta_t]
            u_lat_cmd = -self.K_lat @ (self.x_lat - x_lat_ref)     # [delta_a, delta_r]
            delta_cmd = self.delta_trim + np.array(
                [u_lon_cmd[0], u_lon_cmd[1], u_lat_cmd[0], 0.0],
                dtype=np.float32,
            )

        # 2) Actuator lag, rate limiting, and saturation.
        alpha = self.dt / max(self.act_tau, 1e-4)
        delta_step = alpha * (delta_cmd - self.delta)
        max_delta_step = self.delta_rate_limit * self.dt
        delta_step = np.clip(delta_step, -max_delta_step, max_delta_step)
        self.delta = self.delta + delta_step
        self.delta = np.clip(self.delta, self.delta_min, self.delta_max)
        # Rudder is removed as a control variable and cannot drift through lag.
        self.delta[3] = self.delta_trim[3]

        if self.more_realistic:
            # Nonlinear forces, moments, rigid-body integration and kinematics.
            learner_vel = self._step_realistic_blue()
            theta = float(self.x_lon[3])
            phi = float(self.x_lat[3])
        else:
            # 3) Original reduced-order dynamics integration.
            delta_pert = self.delta - self.delta_trim
            u_lon = np.array([delta_pert[0], delta_pert[1]], dtype=np.float32)
            u_lat = np.array([delta_pert[2], delta_pert[3]], dtype=np.float32)
            # A/B are linearized around trim, so propagate perturbation states.
            xdot_lon = self.A_lon @ (self.x_lon - self.x_lon_trim) + self.B_lon @ u_lon
            xdot_lat = self.A_lat @ (self.x_lat - self.x_lat_trim) + self.B_lat @ u_lat
            self.x_lon = self.x_lon + self.dt * xdot_lon
            self.x_lat = self.x_lat + self.dt * xdot_lat
            self.x_lon = np.clip(self.x_lon, self.x_lon_min, self.x_lon_max)
            self.x_lat = np.clip(self.x_lat, self.x_lat_min, self.x_lat_max)
            # Enforce Eq. 9.60 attitude limits on the actual propagated states.
            self.x_lon[3] = np.clip(self.x_lon[3], -self.theta_cmd_max, self.theta_cmd_max)
            self.x_lat[3] = np.clip(self.x_lat[3], -self.phi_cmd_max, self.phi_cmd_max)

            # 4) Original kinematics: body velocity -> inertial velocity -> position.
            u_b = float(max(self.x_lon[0], 0.1))
            v_b = float(self.x_lat[0])
            w_b = float(self.x_lon[1])
            theta = float(self.x_lon[3])
            phi = float(self.x_lat[3])
            r = float(self.x_lat[2])

            self.learner_psi = float((self.learner_psi + self.dt * r + np.pi) % (2.0 * np.pi) - np.pi)
            R_bi = self._rotation_body_to_inertial(phi, theta, self.learner_psi)
            learner_vel = R_bi @ np.array([u_b, v_b, w_b], dtype=np.float32)
            learner_vel[2] = np.clip(learner_vel[2], -10, 10)
            self.learner_pos = self.learner_pos + self.dt * learner_vel

        # Scripted red drone: aggressive lead pursuit so blue learns to evade.
        los_vec = self.learner_pos - self.ref_pos
        separation = float(np.linalg.norm(los_vec))
        lead_vec = los_vec + self.ref_lead_time * learner_vel
        desired_vec = (1.0 - self.ref_lead_blend) * los_vec + self.ref_lead_blend * lead_vec
        desired_xy_norm = float(np.linalg.norm(desired_vec[:2]))
        if desired_xy_norm < 1e-6:
            desired_psi = self.ref_psi
        else:
            desired_psi = float(np.arctan2(desired_vec[1], desired_vec[0]))

        yaw_err = float((desired_psi - self.ref_psi + np.pi) % (2.0 * np.pi) - np.pi)
        yaw_rate_cmd_raw = self.ref_heading_gain * yaw_err
        yaw_rate_cmd = float(np.clip(yaw_rate_cmd_raw, -self.ref_max_turn_rate, self.ref_max_turn_rate))
        yaw_cmd_saturated = abs(yaw_rate_cmd_raw - yaw_rate_cmd) > 1e-6

        # Adaptive speed keeps the pursuer threatening at multiple ranges.
        desired_speed = self.ref_speed + self.ref_speed_gain * (separation - self.desired_follow_distance)
        speed_cmd = float(np.clip(desired_speed, self.ref_speed_min, self.ref_speed_max))
        vz_cmd = float(np.clip(desired_vec[2], -self.ref_max_climb_rate, self.ref_max_climb_rate))

        # Convert pursuit guidance into a desired inertial velocity. The rigid
        # body must tilt and redistribute four rotor thrusts to achieve it.
        desired_xy_dir = desired_vec[:2] / max(desired_xy_norm, 1e-6)
        desired_velocity = np.array(
            [speed_cmd * desired_xy_dir[0], speed_cmd * desired_xy_dir[1], vz_cmd],
            dtype=np.float32,
        )
        accel_cmd = self.ref_velocity_gain * (desired_velocity - self.ref_vel)
        accel_norm = float(np.linalg.norm(accel_cmd))
        if accel_norm > self.ref_max_accel:
            accel_cmd *= self.ref_max_accel / accel_norm

        # Rate-limit the heading reference; yaw is then achieved through rotor
        # reaction torque rather than being assigned directly.
        self.ref_yaw_cmd = float(
            (self.ref_yaw_cmd + self.dt * yaw_rate_cmd + np.pi) % (2.0 * np.pi) - np.pi
        )
        cpsi_cmd, spsi_cmd = np.cos(self.ref_yaw_cmd), np.sin(self.ref_yaw_cmd)
        force_direction = accel_cmd + np.array([0.0, 0.0, self.gravity], dtype=np.float32)
        force_norm = max(float(np.linalg.norm(force_direction)), 1e-6)
        thrust_direction = force_direction / force_norm
        theta_cmd = float(np.arctan2(
            thrust_direction[0] * cpsi_cmd + thrust_direction[1] * spsi_cmd,
            max(thrust_direction[2], 1e-4),
        ))
        phi_cmd = float(np.arctan2(
            thrust_direction[0] * spsi_cmd - thrust_direction[1] * cpsi_cmd,
            max(thrust_direction[2], 1e-4),
        ))
        theta_cmd = float(np.clip(theta_cmd, -self.ref_max_tilt, self.ref_max_tilt))
        phi_cmd = float(np.clip(phi_cmd, -self.ref_max_tilt, self.ref_max_tilt))
        desired_attitude = np.array([phi_cmd, theta_cmd, self.ref_yaw_cmd], dtype=np.float32)

        substeps = max(self.ref_dyn_substeps, 1)
        dt_ref = self.dt / float(substeps)
        for _ in range(substeps):
            attitude_error = desired_attitude - self.ref_attitude
            attitude_error[2] = (attitude_error[2] + np.pi) % (2.0 * np.pi) - np.pi
            angular_accel_cmd = self.ref_attitude_kp * attitude_error - self.ref_attitude_kd * self.ref_body_rates
            inertia_torque = self.ref_inertia * angular_accel_cmd
            gyro_torque = np.cross(self.ref_body_rates, self.ref_inertia * self.ref_body_rates)
            torque_cmd = np.clip(inertia_torque + gyro_torque, -self.ref_max_torque, self.ref_max_torque)

            collective_cmd = self.ref_mass * force_norm
            mixer = np.array(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [0.0, self.ref_arm_length, 0.0, -self.ref_arm_length],
                    [-self.ref_arm_length, 0.0, self.ref_arm_length, 0.0],
                    [self.ref_yaw_moment_coeff, -self.ref_yaw_moment_coeff,
                     self.ref_yaw_moment_coeff, -self.ref_yaw_moment_coeff],
                ],
                dtype=np.float32,
            )
            rotor_cmd = np.linalg.solve(mixer, np.r_[collective_cmd, torque_cmd]).astype(np.float32)
            rotor_cmd = np.clip(rotor_cmd, 0.0, self.ref_max_rotor_thrust)
            motor_alpha = min(dt_ref / max(self.ref_motor_tau, dt_ref), 1.0)
            self.ref_motor_thrusts += motor_alpha * (rotor_cmd - self.ref_motor_thrusts)

            wrench = mixer @ self.ref_motor_thrusts
            collective_real = float(wrench[0])
            torque_real = wrench[1:]
            angular_accel = (
                torque_real - np.cross(self.ref_body_rates, self.ref_inertia * self.ref_body_rates)
            ) / self.ref_inertia
            self.ref_body_rates += dt_ref * angular_accel

            phi_ref, theta_ref, _ = self.ref_attitude
            p_ref, q_ref, r_ref = self.ref_body_rates
            cos_theta = max(float(np.cos(theta_ref)), 1e-3)
            euler_rates = np.array(
                [
                    p_ref + np.tan(theta_ref) * (q_ref * np.sin(phi_ref) + r_ref * np.cos(phi_ref)),
                    q_ref * np.cos(phi_ref) - r_ref * np.sin(phi_ref),
                    (q_ref * np.sin(phi_ref) + r_ref * np.cos(phi_ref)) / cos_theta,
                ],
                dtype=np.float32,
            )
            self.ref_attitude += dt_ref * euler_rates
            self.ref_attitude[0:2] = np.clip(self.ref_attitude[0:2], -np.deg2rad(80.0), np.deg2rad(80.0))
            self.ref_attitude[2] = (self.ref_attitude[2] + np.pi) % (2.0 * np.pi) - np.pi

            rotation_ref = self._rotation_body_to_inertial(*self.ref_attitude)
            thrust_inertial = rotation_ref[:, 2] * collective_real
            speed_3d = float(np.linalg.norm(self.ref_vel))
            drag_force = -self.ref_linear_drag * self.ref_vel - self.ref_quadratic_drag * speed_3d * self.ref_vel
            acceleration = (
                thrust_inertial + drag_force
            ) / self.ref_mass - np.array([0.0, 0.0, self.gravity], dtype=np.float32)
            self.ref_vel += dt_ref * acceleration
            self.ref_pos += dt_ref * self.ref_vel

        self.ref_psi = float(self.ref_attitude[2])
        speed_real = float(np.linalg.norm(self.ref_vel[:2]))
        vz_real = float(self.ref_vel[2])
        yaw_rate_real = float(self.ref_body_rates[2])
        yaw_real_saturated = bool(
            abs(yaw_rate_real) > 1.2 * self.ref_max_turn_rate
            or np.any(
                (self.ref_motor_thrusts < 1e-4)
                | (self.ref_motor_thrusts > self.ref_max_rotor_thrust - 1e-4)
            )
        )

        # Evasion metrics.
        rel_vec = self.learner_pos - self.ref_pos
        separation = float(np.linalg.norm(rel_vec))
        rel_dir = rel_vec / max(separation, 1e-6)
        opening_speed = float(np.dot(learner_vel - self.ref_vel, rel_dir))

        out_of_bounds = bool(
            self.terminate_on_world_limit
            and np.isfinite(self.world_limit)
            and (
                np.any(np.abs(self.learner_pos) > self.world_limit)
                or np.any(np.abs(self.ref_pos) > self.world_limit)
            )
        )
        too_close = separation < self.min_separation
        terminated = bool(out_of_bounds or too_close)
        truncated = bool(self._episode_len >= self.max_steps)

        # Reward: stay alive, increase separation, and avoid jitter.
        reward = 0.25
        reward += 0.02 * min(separation, 100.0)
        reward += 0.05 * opening_speed
        reward -= self.action_change_penalty * float(np.linalg.norm(a - last_action))
        # Penalize rapid roll/pitch motion directly. Action smoothing alone can
        # still permit a policy to sustain oscillations with alternating inputs.
        reward -= self.body_rate_penalty * float(self.x_lat[1] ** 2 + self.x_lon[2] ** 2)
        if too_close:  # kamikaze intercepted evader
            reward -= 20.0
        if out_of_bounds:
            reward -= 10.0
        if truncated and not terminated:  # evader survived full horizon
            reward += 10.0

        info = {
            "separation": separation,
            "position_error_norm": separation,
            "velocity_error_norm": float(np.linalg.norm(learner_vel - self.ref_vel)),
            "too_close": too_close,
            "out_of_bounds": out_of_bounds,
            "caught": too_close,
            "learner_color": "blue",
            "reference_color": "red",
            "telemetry_red_los_angle_error_rad": yaw_err,
            "telemetry_red_yaw_error_rad": yaw_err,
            "telemetry_red_r_cmd_rad_s": yaw_rate_cmd,
            "telemetry_red_r_real_rad_s": yaw_rate_real,
            "telemetry_red_speed_cmd_mps": speed_cmd,
            "telemetry_red_speed_mps": speed_real,
            "telemetry_opening_speed_mps": opening_speed,
            "telemetry_red_cmd_turn_saturated": yaw_cmd_saturated,
            "telemetry_red_real_turn_saturated": yaw_real_saturated,
            "control_mode": self.control_mode,
            "blue_dynamics_mode": "ar3_6dof" if self.more_realistic else "linear_reduced_order",
            "learner_phi_rad": phi,
            "learner_theta_rad": theta,
            "learner_psi_rad": self.learner_psi,
            "reference_phi_rad": float(self.ref_attitude[0]),
            "reference_theta_rad": float(self.ref_attitude[1]),
            "reference_psi_rad": self.ref_psi,
            "telemetry_red_collective_thrust_n": float(np.sum(self.ref_motor_thrusts)),
            "telemetry_red_motor_thrusts_n": self.ref_motor_thrusts.copy(),
            "delta_e": float(self.delta[0]),
            "delta_t": float(self.delta[1]),
            "delta_a": float(self.delta[2]),
            "delta_r": float(self.delta[3]),  # fixed at trim; retained for telemetry compatibility
        }
        info.update(self.blue_last_telemetry)

        return self._get_obs(), float(reward), terminated, truncated, info
