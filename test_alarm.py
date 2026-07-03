import pygame

pygame.mixer.init()

pygame.mixer.music.load("sounds/alarm.wav")
pygame.mixer.music.play()

print("Playing Alarm...")

input("Press Enter to Exit")