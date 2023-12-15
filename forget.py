"""
Responsibility:
    Mimics human memory patterns and allows customization of character
    memory traits.
Process:
    Manages the forgetting of memories when not in use, simulating
    human-like memory patterns for the virtual characters.
"""

import math

def memory_retention(time_elapsed,
                     memory_stability,
                     decay_constant,
                     boost_factor,
                     last_access):
    """
    Summary:
    Calculate memory retention and update memory stability based on the
    forgetting model.

    Args:
    time_elapsed (float):
        Elapsed time since the last access to the memory.
    memory_stability (float):
        Memory stability, determining how "strong" a memory is.
    decay_constant (float):
        Decay constant that defines how forgetful a character is in
        general.
    boost_factor (float):
        Determines how fast memories are strengthened through
        repetition.
    last_access (float):
        Time of the last access to the memory.

    Returns:
    retention (float):
        The memory retention value.
    stability (float):
        The updated memory stability based on the boost factor.

    Reference:
    The forgetting model is based on the paper "Memories for Virtual AI
    Characters" by Fabian Landwehr, Erika Varis Doggett, and Romann M.
    Weber (Sept. 2023), p.242.
    """
    retention = math.exp(-decay_constant * time_elapsed / memory_stability)
    # Update stability with boost factor
    stability = memory_stability * boost_factor
    return retention, stability

