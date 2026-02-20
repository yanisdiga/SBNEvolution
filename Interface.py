import pygame

# Fonction permettant d'afficher le bandeau en haut de l'écran avec les informations de la simulation
def draw_dashboard(screen, clock, agents, total_steps, params, width, dashboard_size, font):
    # 1. Dessin du fond du bandeau
    pygame.draw.rect(screen, (20, 20, 20), (0, 0, width, dashboard_size-5)) # -5 pour laisser un petit gap
    pygame.draw.line(screen, (150, 150, 150), (0, dashboard_size-5), (width, dashboard_size-5), 2) # -5 pour laisser un petit gap

    # 2. Informations à afficher
    fps = int(clock.get_fps())
    pop = len(agents)
    mode_txt = "Photosynthèse" if params["MODE_FOOD"] == 1 else "Alimentation"
    
    # Rendu des textes
    txt_test = font.render(f"TEST: {params['TEST_NAME']}", True, (255, 255, 255))
    txt_pop  = font.render(f"POPULATION: {pop}", True, (0, 255, 100) if pop > 0 else (255, 50, 50))
    txt_step = font.render(f"STEPS: {total_steps}", True, (200, 200, 200))
    txt_mode = font.render(f"MODE: {mode_txt}", True, (100, 200, 255))
    txt_fps  = font.render(f"FPS: {fps}", True, (255, 255, 0))

    # 3. Positionnement sur le bandeau
    screen.blit(txt_test, (20, 15))
    screen.blit(txt_mode, (300, 15))
    screen.blit(txt_step, (550, 15))
    screen.blit(txt_pop,  (750, 15))
    screen.blit(txt_fps,  (width - 100, 15))