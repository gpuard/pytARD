import numpy as np

class SimulationParameters:
    def __init__(
        self,
        max_wave_frequency      = 2e4,
        simulation_time         = 1,
        samples_per_wave_length = 7,
        c                       = 343,
        samples_per_second      = 44100,
        enable_multicore        = True,
        auralize                = None,
        verbose                 = False
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
        max_wave_frequency : float
            Uppermost frequency of simulation. Can be dialed in lower to enhance performance.
        simulation_time : float
            Simulation time [s].
        samples_per_wave_length : int
            Number of spatial samples per wave length. Usually 2 to 4. Lower values decrease 
            resolution but enhances performance.
        c : float
            Speed of sound [m/s]. Depends on air temperature, pressure and humidity. 
        samples_per_second : int
            Sampling rate. The higher, the more fidelity but lower performance.
        enable_multicore : bool
            Enables performance optimization by enabling multicore/multi-CPU processing.
        auralize : ndarray
            Auralizes (= makes hearable) the room by creating an impulse response (IR).
            Format is a list with mic positions. If array is empty, no auralization is being made.
        verbose : boolean
            Prints information on the terminal for debugging and status purposes.
        '''

        self.max_wave_frequency = max_wave_frequency
        self.c = c
        self.samples_per_second = samples_per_second
        self.simulation_time = simulation_time
        self.samples_per_wave_length = samples_per_wave_length
        
        self.enable_multicore = enable_multicore
        self.auralize = auralize
        self.verbose = verbose
        
        # grid spacing
        # the Nyquist sampling theorem
        min_wave_length = self.c / max_wave_frequency
        self.dx = min_wave_length / samples_per_wave_length
        self.dy = self.dx
        
        # Calculating the number of samples the simulation takes.
        self.number_of_time_samples = int(simulation_time * samples_per_second)

        # Calculate time stepping (Δ_t)
        self.delta_t = simulation_time / self.number_of_time_samples
        
        self.delta_t = min(SimulationParameters.get_dt_clf_condition(self.dx, self.c), self.delta_t)

        if verbose:
            print(f"Insantiated simulation.\nNumber of samples: {self.number_of_time_samples} | Δ_t: {self.delta_t}")

    @staticmethod
    def calculate_voxelization_step(c, samples_per_wave_length, max_simulation_frequency):
        '''
        Calculate voxelization step for the segmentation of the room (voxelizing the scene).
        The cell size is fixed by the chosen wave length and the number of spatial samples per 
        wave length.
        Calculation: ℎ <= 𝑐 / (2 * 𝑓_max)

        Parameters
        ----------
        samples_per_wave_length : int
            Number of spatial samples per wave length. Usually 2 to 4.

        Returns
        -------
        float
            ℎ, the voxelization step. In numerics and papers, it's usually referred to ℎ. 
        '''
        return c / (samples_per_wave_length * max_simulation_frequency)
 
    @staticmethod    
    def get_dt_clf_condition(dx, speedOfsound):
        '''
        Courant–Friedrichs–Lewy condition - is a necessary condition for convergence while solving  hyperbolic PDEs numerically.
        The constant is choosen as described in "An efficient GPU-based time domain solver for the acoustic wave equation".
        
        https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition

        Parameters
        ----------
        dx : Type
            Grid spacing.
        speedOsamples_per_secondound : FLOAT
            Speed of sound.
    
        Returns
        -------
        BOOLEAN
            Returns the maximal possible time stepping which fulfills the conditing.

        '''
        # TODO the choice constant being used should be explained
        return dx / speedOfsound * np.sqrt(3) - 1e-10