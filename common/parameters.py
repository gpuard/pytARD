


class SimulationParameters:
    def __init__(
        self,
        max_simulation_frequency,
        T,
        spatial_samples_per_wave_length=6,
        c=343,
        Fs=8000,
        normalization_constant=1,
        auralize=None,
        verbose=False,
        visualize=False
    ):
        '''
        Parameter container class for ARD simulation. Contains all relevant data to instantiate
        and run ARD simulator.

        Parameters
        ----------
        partitions : ndarray
            Collection of 1D, 2D or 3D partitions (rooms). Array of PartitionData objects.
        src_pos : ndarray
            Location of signal source inside the room. Can be 1D, 2D or 3D.
        max_simulation_frequency : float
            Uppermost frequency of simulation. Can be dialed in lower to enhance performance.
        T : float
            Simulation time [s].
        spatial_samples_per_wave_length : int
            Number of spatial samples per wave length. Usually 2 to 4. Lower values decrease 
            resolution but enhances performance.
        c : float
            Speed of sound [m/s]. Depends on air temperature, pressure and humidity. 
        Fs : int
            Sampling rate. The higher, the more fidelity but lower performance.
        auralize : ndarray
            Auralizes (= makes hearable) the room by creating an impulse response (IR).
            Format is a list with mic positions. If array is empty, no auralization is being made.
        verbose : boolean
            Prints information on the terminal for debugging and status purposes.
        visualize : boolean
            Visualizes wave propagation in a plot.
        '''

        assert(T > 0), "Error: Simulation duration must be a positive number greater than 0."
        assert(Fs > 0), "Error: Sample rate must be a positive number."
        assert(c > 0), "Error: Speed of sound must be a positive number."
        assert(max_simulation_frequency >
               0), "Error: Uppermost frequency of simulation must be a positive number."
        assert(max_simulation_frequency > (Fs / 2), "Nyquist-Shannon theorem violated. Make sure upper frequency limit is Fs / 2.")

        self.max_simulation_frequency = max_simulation_frequency
        self.c = c
        self.Fs = Fs
        self.T = T
        self.spatial_samples_per_wave_length = spatial_samples_per_wave_length

        # Calculating the number of samples the simulation takes.
        self.number_of_samples = int(T * Fs)

        # Calculate time stepping (Δ_t)
        self.delta_t = T / self.number_of_samples

        self.normalization_constant = normalization_constant
        self.auralize = auralize
        self.verbose = verbose
        self.visualize = visualize

        if verbose:
            print(f"Insantiated simulation.\nNumber of samples: {self.number_of_samples} | Δ_t: {self.delta_t}")

    @staticmethod
    def calculate_voxelization_step(sim_param):
        '''
        Calculate voxelization step for the segmentation of the room (voxelizing the scene).
        The cell size is fixed by the chosen wave length and the number of spatial samples per 
        wave length.
        Calculation: ℎ <= 𝑐 / (2 * 𝑓_max)

        Parameters
        ----------
        sim_param : SimulationParameters
            ARD simulation parameter object

        Returns
        -------
        float
            ℎ, the voxelization step. In numerics and papers, it's usually referred to ℎ. 
        '''
        return sim_param.c / (sim_param.spatial_samples_per_wave_length * sim_param.max_simulation_frequency)

