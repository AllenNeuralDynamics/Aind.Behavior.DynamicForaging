import numpy as np
from typing import Literal, Optional


class TrialGenerator():
    def __init___(*args, **kwargs):
        """
        Docstring for __init___
        
        :param args: Description
        :param kwargs: Description
        """

    def generate_trial(*args, **kwargs):
        """
        Docstring for generate_trial
        
        :param args: Description
        :param kwargs: Description
        """

        raise NotImplementedError()
    
BlockBehaviorEvaluationMode = Literal[
                                        "ignore",   # do not take behavior into account when switching blocks
                                        "end",      # behavior must be stable at end of block to allow switching
                                        "anytime"]  # behavior can be stable anytime in block to allow switching


class TrialOutcome(BaseModel):
    """
        Info from previous trial needed to generate next trial
    """

    trial_in_block: int
    block_length: int



class CoupledTrialGenerator(TrialGenerator):

    def __init___(self, trial_outcome: TrialOutcome):
        """
        Check if block should switch, generate next block if necessary, and  generate next trial
        
        :param trial_outcome: outcome of previous trial
        """

        switch_block = self.check_block_transition(trial_outcome)

        if switch_block:
            next_block = self.generate_block()
        
        self.generate_trial(trial_outcome, next_block)


    def check_block_behavior(self, choice_history: np.ndarray, 
                             right_reward_prob: float,
                             left_reward_prob: float,
                             beh_eval_mode:BlockBehaviorEvaluationMode,  
                             block_length:int, 
                             points_in_a_row: int=3, 
                             switch_thr: float=0.8, 
                             kernel_size: int = 2) -> Optional[bool]: 
        """
        This function replaces _check_advanced_block_switch. Checks if behavior within block 
        allows for switching
    
        choice_history: 2D array (rows = sides, columns = trials) with 0: left, 1: right and 2: ignored entries.
        right_reward_prob: reward probability for right side 
        left_reward_prob: reward probability for left side
        beh_eval_mode: mode to evaluate behavior
        block_length: planned number of trials in current block. In couple trials, both sides have same block length so block length is int. 
        points_in_a_row: number of consecutive trials above threshold required
        switch_thr: fraction threshold to define stable behavior
        kernel_size: kernal to evaluate choice fraction

        """
        current_trial_n = choice_history.shape[1]
        # do not prohibit block transition if does not rely on behavior or not enough trials to evaluate  or reward probs are the same.
        if beh_eval_mode == "ignore" or left_reward_prob == right_reward_prob or current_trial_n < kernel_size:   
            return True                 
               
        # compute fraction of right choices with running average using a sliding window
        choice_fraction = np.empty(current_trial_n - kernel_size + 1, dtype=float)  # create empty array to store running averages
        for i in range(current_trial_n - kernel_size + 1):
            window = choice_history[i:i+kernel_size].astype(float)
            window[window == 2] = np.nan
            choice_fraction[i] = np.nanmean(window)

        if block_length > len(choice_fraction): 
            return True   # not enough trials to evaluate so don't prohibit switch
        
        block_choice_frac = choice_fraction[-block_length:]
        delta = abs((left_reward_prob - right_reward_prob) * float(switch_thr))    # margin based on right and left probabilities and scaled by switch threshold. Window for evaluating behavior
        threshold = [0, left_reward_prob - delta] if left_reward_prob > right_reward_prob else [left_reward_prob + delta, 1]

        above_threshold_pts = (block_choice_frac >= threshold[0]) & (block_choice_frac <= threshold[1]) # block_choice_fractions above threshold 
        consecutive_lengths, consecutive_indices = self.consecutive_length(above_threshold_pts, 1)  

        if consecutive_lengths.size == 0:
            return False
        
        qualifying_runs = consecutive_indices[consecutive_lengths > points_in_a_row] # runs where consecutive lenghts is above check
        
        if beh_eval_mode == "end":  # requires consecutive trials ending on the last trial
            # check if the current trial occurs at the end of a long enough consecutive run above threshold
            last_threshold_trial = len(above_threshold_pts) - 1
            return last_threshold_trial in qualifying_runs[:, 1]    # if the last threshold trial is with in the last set of consecutive trials 
            
        elif beh_eval_mode == "anytime":
            return np.any(qualifying_runs)
        

    @staticmethod
    def consecutive_length(arr, target=1) -> tuple[np.ndarray, np.ndarray]:
        """
            Get consecutive lengths and start/end indices of a target value in arr.

            :returns consecutive_lengths: array of array of integers where each number is the length of a consecutive run of trials above threshold
                     consecutive_indices: array of shape (n_runs, 2) where each row is [start_index, end_index] of consecutive run trials

        """
        arr = np.asarray(arr) == target  # convert to boolean array
        if not np.any(arr):
            return np.array([]), np.array([])

        # Find where value changes (diff != 0)
        diff = np.diff(arr.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0]

        # Handle edge cases: starts at 0 or ends at last index
        if arr[0]:
            starts = np.r_[0, starts]
        if arr[-1]:
            ends = np.r_[ends, len(arr) - 1]

        lengths = ends - starts + 1
        indices = np.vstack([starts, ends]).T
        return lengths, indices


    def check_block_transition(self, 
                             min_block_reward: int,
                             block_left_rewards: int,
                             block_right_rewards: int,
                             block_length_history: np.ndarray,
                             choice_history: np.ndarray, 
                             right_reward_prob: float,
                             left_reward_prob: float,
                             beh_eval_mode:BlockBehaviorEvaluationMode,  
                             block_length:int, 
                             points_in_a_row: int=3, 
                             switch_thr: float=0.8, 
                             kernel_size: int = 2) -> bool:
        """
        min_block_reward: minimum reward to allow switching
        block_left_rewards: number of left rewarded trials in current block
        block_right_rewards: number of left rewarded trials in current block
        block_length_history: 1D array of all block length. 1D for coupled
        choice_history: 2D array (rows = sides, columns = trials) with 0: left, 1: right and 2: ignored entries.
        right_reward_prob: reward probability for right side 
        left_reward_prob: reward probability for left side
        beh_eval_mode: mode to evaluate behavior
        block_length: planned number of trials in current block. In couple trials, both sides have same block length so block length is int. 
        points_in_a_row: number of consecutive trials above threshold required
        switch_thr: fraction threshold to define stable behavior
        kernel_size: kernal to evaluate choice fraction
        """
            
        current_trial_n = choice_history.shape[1]

        # first trial of the current block
        first_trial_of_block = current_trial_n == sum(block_length_history[:-1])

        # has planned block length been reached?
        block_length_met = current_trial_n + 1 >= sum(block_length_history)

        # is behavior qualified to switch?
        behavior_ok = self.check_block_behavior(
            choice_history,
            right_reward_prob,
            left_reward_prob,
            beh_eval_mode,
            block_length,
            points_in_a_row,
            switch_thr,
            kernel_size,
        )

        # conditions to switch: 
        #   - planned block length reached 
        #   - not first trial of the block
        #   - minimum reward requirement is reached 
        #   - behavior is stable 
        return first_trial_of_block and block_length_met and (
            block_left_rewards + block_right_rewards < min_block_reward or behavior_ok
        )
        

    def generate_block():
        """
        Docstring for generate_block
        """

    def generate_trial(*args, **kwargs):
        """
        Docstring for generate_trial
        
        :param args: Description
        :param kwargs: Description
        """

        raise NotImplementedError() 