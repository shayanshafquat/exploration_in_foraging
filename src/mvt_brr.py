import numpy as np
import matplotlib.pyplot as plt

class MVTModel:
    def __init__(self, decay_type='exponential', decay_rate=None):
        # Initialize the model with a decay type, which affects how rewards decay over time.
        self.decay_type = decay_type
        self.decay_rate = decay_rate if decay_rate else None
        # Initialize model parameters based on the decay type.
        self.TravelTs, self.reso, self.A, self.a = self.initialize_parameters()
        # Number of patch types (ecological contexts or areas).
        self.NrPatchTypes = len(self.a)

    def initialize_parameters(self):
        # Travel times to patches (in abstract units).
        TravelTs = np.array([6])
        # Initial yields of different patch types (abstract units).
        A = np.array([32.5, 45, 57.5])
        # Different parameters are set based on decay type.
        if self.decay_type == 'exponential':
            reso = 50  # Resolution (time length).
            a = np.array([0.075, 0.075, 0.075])  # Decay rates for each patch type.
        else:
            a = np.array([self.decay_rate, self.decay_rate, self.decay_rate])
            reso = 120
        return TravelTs, reso, A, a

    def calculate_rewards(self, A, a, T):
        # Calculate the reward, reward rate, and gain based on decay type.
        if self.decay_type == 'exponential':
            Reward = np.maximum(A * np.exp(-a * T), 0)
            RRE = -a * A * np.exp(-a * T)
            Gain = (A / a) * (1 - np.exp(-a * T))
        elif self.decay_type == 'linear':
            Reward = np.maximum(A - a * T, 0)
            RRE = -a if A - a * T > 0 else 0  # Adjusted to handle zero reward case
            Gain = np.maximum(A * T - 0.5 * a * T**2, 0)  # Ensure Gain is non-negative
        else:
            raise ValueError("Invalid decay type. Use 'exponential' or 'linear'.")
        return Reward, RRE, Gain

    def calculate_optimal_times(self, AllGain):
        # Calculate optimal leaving times considering all gain arrays and travel times.
        multiPatchRR = np.zeros((self.reso, len(self.TravelTs), self.NrPatchTypes))
        Tleave = np.zeros((len(self.TravelTs), self.NrPatchTypes), dtype=int)
        RRleave = np.zeros((len(self.TravelTs), self.NrPatchTypes))
        
        for P in range(self.NrPatchTypes):
            for Tind, TravelT in enumerate(self.TravelTs):
                RR = np.zeros(self.reso)
                for T in range(1, self.reso + 1):
                    RR[T - 1] = AllGain[T - 1, P] / (TravelT + T)
                    multiPatchRR[T - 1, Tind, P] = RR[T - 1]
                Tleave[Tind, P] = np.argmax(RR)
                RRleave[Tind, P] = np.max(RR)
        return multiPatchRR, Tleave, RRleave

    # Compute overall reward rates
    def compute_overall_rr(self, multiPatchRR, pPatch, AllGainE, AllRewardE):
        NrPatchTypes = len(pPatch)
        OverallRR = np.zeros((self.reso, self.reso, self.reso, len(self.TravelTs)))
        maxRR = np.zeros(len(self.TravelTs))
        Tmax = np.zeros((len(self.TravelTs), 3), dtype=int)
        GainEmax = np.zeros((len(self.TravelTs), 3))
        RewardEmax = np.zeros((len(self.TravelTs), 3))

        for Tind, TravelT in enumerate(self.TravelTs):
            Timesteps = np.arange(1, self.reso + 1) + TravelT  # Adjust time steps
            T1 = np.tile(Timesteps[:, np.newaxis, np.newaxis], (1, self.reso, self.reso))  # Extend and tile across dimensions
            T2 = np.tile(Timesteps[np.newaxis, :, np.newaxis], (self.reso, 1, self.reso))  # Extend and tile across dimensions
            T3 = np.tile(Timesteps[np.newaxis, np.newaxis, :], (self.reso, self.reso, 1))  # Extend and tile across dimensions

            # Compute proportions
            denominator = T1 * pPatch[0] + T2 * pPatch[1] + T3 * pPatch[2]
            proportionT1 = T1 * pPatch[0] / denominator
            proportionT2 = T2 * pPatch[1] / denominator
            proportionT3 = T3 * pPatch[2] / denominator

            # Replicate multiPatchRR across dimensions
            curPatchRR = np.zeros((self.reso, self.reso, self.reso, 3))
            curPatchRR[:, :, :, 0] = np.tile(multiPatchRR[:, Tind, 0][:, np.newaxis, np.newaxis], (1, self.reso, self.reso))
            curPatchRR[:, :, :, 1] = np.tile(multiPatchRR[:, Tind, 1][np.newaxis, :, np.newaxis], (self.reso, 1, self.reso))
            curPatchRR[:, :, :, 2] = np.tile(multiPatchRR[:, Tind, 2][np.newaxis, np.newaxis, :], (self.reso, self.reso, 1))

            # Calculate the overall reward rate
            OverallRR[:, :, :, Tind] = curPatchRR[:, :, :, 0] * proportionT1 \
                                    + curPatchRR[:, :, :, 1] * proportionT2 \
                                    + curPatchRR[:, :, :, 2] * proportionT3

            # Find the maximum reward rate and its index
            maxRR[Tind] = np.max(OverallRR[:, :, :, Tind])
            idx_max = np.argmax(OverallRR[:, :, :, Tind])
            Tmax[Tind] = np.unravel_index(idx_max, (self.reso, self.reso, self.reso))

            # Retrieve the maximum gain and reward
            GainEmax[Tind, :] = [AllGainE[Tmax[Tind][i], i] for i in range(3)]
            RewardEmax[Tind, :] = [AllRewardE[Tmax[Tind][i], i] for i in range(3)]

            # # Output results
            # print("Max Reward Rates:", maxRR)
            # print("Optimal Times (indices):", Tmax+1)
            # print("Gain at Max:", GainEmax)
            # print("Reward at Max:", RewardEmax)

        return maxRR, Tmax[0]+1, GainEmax, RewardEmax

    def run(self):
        # Main method to calculate rewards and optimal times for both rich and poor environments.
        AllRewardE = np.zeros((self.reso, self.NrPatchTypes))
        AllRRE = np.zeros((self.reso, self.NrPatchTypes))
        AllGainE = np.zeros((self.reso, self.NrPatchTypes))
        
        for P in range(self.NrPatchTypes):
            for T in range(1, self.reso + 1):
                RewardE, RRE, GainE = self.calculate_rewards(self.A[P], self.a[P], T)
                AllRewardE[T - 1, P] = RewardE
                AllRRE[T - 1, P] = RRE
                AllGainE[T - 1, P] = GainE

        rich_proportions = np.array([0.2, 0.3, 0.5])
        poor_proportions = np.array([0.5, 0.3, 0.2])
        multiPatchRR_rich, Tleave_rich, RRleave_rich = self.calculate_optimal_times(AllGainE)
        maxRR_rich, Tmax_rich, GainEmax_rich, RewardEmax_rich = self.compute_overall_rr(multiPatchRR_rich, rich_proportions, AllGainE, AllRewardE)
        multiPatchRR_poor, Tleave_poor, RRleave_poor = self.calculate_optimal_times(AllGainE)
        maxRR_poor, Tmax_poor, GainEmax_poor, RewardEmax_poor = self.compute_overall_rr(multiPatchRR_poor, poor_proportions, AllGainE, AllRewardE)
        # self.plot_results(Tmax_rich, Tmax_poor)
        return Tmax_rich, Tmax_poor

    def get_average_reward_rate(self, proportions):
        """
        Calculate the average reward rate for a given environment defined by patch proportions.

        Parameters:
        - proportions: array-like, the proportions of each patch type in the environment.

        Returns:
        - average_reward_rate: float, the average reward rate for the environment.
        """
        # Calculate rewards and gains for all patch types
        AllRewardE = np.zeros((self.reso, self.NrPatchTypes))
        AllGainE = np.zeros((self.reso, self.NrPatchTypes))

        for P in range(self.NrPatchTypes):
            for T in range(1, self.reso + 1):
                RewardE, _, GainE = self.calculate_rewards(self.A[P], self.a[P], T)
                AllRewardE[T - 1, P] = RewardE
                AllGainE[T - 1, P] = GainE

        # Calculate optimal times and reward rates for the environment
        multiPatchRR, _, _ = self.calculate_optimal_times(AllGainE)

        # Compute overall reward rates for the given proportions
        maxRR, _, _, _ = self.compute_overall_rr(multiPatchRR, proportions, AllGainE, AllRewardE)

        # Return the average reward rate, which is maxRR in this context
        return maxRR[0]
    
    def plot_results(self, Tmax_rich, Tmax_poor):
        # Plot the results for optimal leaving times in different environments.
        patch_types = ['Low', 'Med', 'High']
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(patch_types, Tmax_rich + 1, marker='o', label='Rich Environment')
        ax.plot(patch_types, Tmax_poor + 1, marker='o', label='Poor Environment')
        ax.set_xlabel('Patch Types')
        ax.set_ylabel('Optimal Leaving Time')
        ax.set_title(f'Optimal Leaving Times for Decay Type: {self.decay_type}')
        ax.legend()
        plt.show()

# # Instantiate and run the model
# model = MVTModel(decay_type='exponential')
# model.run()