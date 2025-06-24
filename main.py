# main.py

import cv2
import pygame
from gesture.predict import GestureRecognizer
from game.tetris import Tetris

# Inizializza Pygame
pygame.init()

# Inizializza la webcam
cam_width, cam_height = 600, 480
cam = cv2.VideoCapture(0)
cam.set(3, cam_width)
cam.set(4, cam_height)

# Inizializza Pygame display accanto alla webcam
WIN_WIDTH = cam_width + 300
WIN_HEIGHT = cam_height + 120
screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Tetris Gesture Control")

# Crea oggetti
tetris = Tetris()
recognizer = GestureRecognizer()

# Mapping gesto → evento pygame
GESTURE_KEYMAP = {
    "SINISTRA": pygame.K_LEFT,
    "DESTRA": pygame.K_RIGHT,
    "GIÙ": pygame.K_DOWN,
    "pugno": ord('p'),
    "Ok": ord('q'),
    "L": ord('e')
}

font = pygame.font.SysFont("Arial", 24)

def convert_cv_to_pygame(cv_image):
    """Converti immagine OpenCV (BGR) in surface Pygame"""
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    cv_image = cv2.flip(cv_image, 1)
    return pygame.image.frombuffer(cv_image.tobytes(), cv_image.shape[1::-1], "RGB")

def draw_overlay(pygame_frame, gesture_name):
    """Disegna overlay con il nome del gesto rilevato"""
    if gesture_name:
        text_surface = font.render(f"Gesto: {gesture_name.upper()}", True, (255, 255, 255))
        pygame_frame.blit(text_surface, (10, 10))

def main():
    running = True
    while running:
        success, frame = cam.read()
        if not success:
            break

        frame, gesture = recognizer.recognize(frame)

        # Converti per Pygame
        webcam_surface = convert_cv_to_pygame(frame)

        # Gestione eventi da gesto
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False

        # Se c'è un gesto valido, crea evento fittizio
        if gesture in GESTURE_KEYMAP:
            key_event = pygame.event.Event(pygame.KEYDOWN, key=GESTURE_KEYMAP[gesture])
            events.append(key_event)

        # Esegui un frame del gioco
        tetris.run_frame(events)
        tetris.draw()

        # Disegna tutto
        screen.fill((0, 0, 0))
        screen.blit(webcam_surface, (0, 0))
        screen.blit(tetris.screen, (cam_width, 0))
        draw_overlay(screen, gesture)
        pygame.display.flip()

        tetris.tick()

    cam.release()
    pygame.quit()

if __name__ == "__main__":
    main()