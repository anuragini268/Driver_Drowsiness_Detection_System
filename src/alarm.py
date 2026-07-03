import winsound

def play_alarm():
    """
    Plays the default Windows beep sound.
    """
    winsound.Beep(2500, 1000)   # Frequency = 2500 Hz, Duration = 1000 ms