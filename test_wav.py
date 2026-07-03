import wave

try:
    wav = wave.open("sounds/alarm.wav", "rb")
    print("WAV File is OK")
    wav.close()
except Exception as e:
    print(e)