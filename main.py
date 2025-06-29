# main.py

import cv2
import pygame
from gesture.predict import GestureRecognizer
from game.tetris import Tetris

# Inizializza Pygame
pygame.init()

# Inizializza la webcam
cam = cv2.VideoCapture(0)

# Dimensioni native della webcam
native_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
native_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

if native_width == 0 or native_height == 0:
    native_width, native_height = 640, 480

cam_width = native_width
cam_height = native_height

TETRIS_WIDTH = 300
TETRIS_HEIGHT = 600

win_width = max(cam_width + 300, 940)
win_height = max(cam_height, 600)

# Ottieni risoluzione dello schermo
screen_info = pygame.display.Info()
screen_width = screen_info.current_w
screen_height = screen_info.current_h

# Scala la finestra se troppo grande
MAX_SCALE = 0.80
if win_width > screen_width or win_height > screen_height:
    scale = min(
        screen_width / win_width,
        screen_height / win_height,
        MAX_SCALE
    )
    cam_width = int(native_width * scale)
    cam_height = int(native_height * scale)
    TETRIS_WIDTH = int(TETRIS_WIDTH * scale)
    TETRIS_HEIGHT = int(TETRIS_HEIGHT * scale)
    win_width = native_width + TETRIS_WIDTH
    win_height = max(native_height, TETRIS_HEIGHT)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

# Inizializza Pygame display accanto alla webcam
screen = pygame.display.set_mode((win_width, win_height))
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

        # Barra info
        info_surface = pygame.Surface((TETRIS_WIDTH, 130))
        tetris.draw_info_bar(info_surface)

        # Calcola gli offset di cam e tetris
        cam_offset = (win_height - cam_height) // 2
        tetris_offset = (win_height - (TETRIS_HEIGHT + 130)) // 2

        # Disegna tutto
        screen.fill((0, 0, 0))
        screen.blit(webcam_surface, (0, cam_offset))
        screen.blit(tetris.screen, (cam_width, tetris_offset))
        screen.blit(info_surface, (cam_width, tetris_offset + TETRIS_HEIGHT))
        draw_overlay(screen, gesture)
        pygame.display.flip()

        tetris.tick()

    cam.release()
    pygame.quit()

if __name__ == "__main__":
    main()